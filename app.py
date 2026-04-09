#!/usr/bin/env python3

import os
import sys
import platform
import json
import uuid
import threading
import subprocess
import socket
import base64
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory, Response
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Cache-Control'] = 'no-store, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/favicon.ico')
def favicon():
    return Response(status=204)


# ─────────────────────────────────────────────
# SYSTEM UTILITIES
# ─────────────────────────────────────────────

def _find_executable(name):
    if os.path.isabs(name) and os.path.exists(name):
        return name
    for path_dir in os.environ.get('PATH', '').split(os.pathsep):
        full = os.path.join(path_dir, name)
        if os.path.exists(full) and os.access(full, os.X_OK):
            return full
        if platform.system() == 'Windows':
            fullexe = full + '.exe'
            if os.path.exists(fullexe) and os.access(fullexe, os.X_OK):
                return fullexe
    return None


def ffmpeg_cmd():
    return os.environ.get('FFMPEG_PATH') or _find_executable('ffmpeg') or 'ffmpeg'


def ffprobe_cmd():
    return os.environ.get('FFPROBE_PATH') or _find_executable('ffprobe') or 'ffprobe'


def ffmpeg_install_hint():
    system = platform.system()
    if system == 'Windows':
        return 'Install ffmpeg from https://ffmpeg.org/download.html and add the bin folder to PATH.'
    if system == 'Darwin':
        return 'Install ffmpeg with Homebrew: brew install ffmpeg.'
    return 'Install ffmpeg via your package manager, e.g., apt install ffmpeg.'


def get_server_api_key() -> str:
    return os.environ.get('OPENAI_API_KEY', '').strip()


def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return False
        except OSError:
            return True


UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────
# OPENAI CLIENT CACHE
# ─────────────────────────────────────────────

_openai_clients: dict = {}


def _openai_client(api_key: str) -> OpenAI:
    if api_key not in _openai_clients:
        _openai_clients[api_key] = OpenAI(api_key=api_key)
    return _openai_clients[api_key]


# ─────────────────────────────────────────────
# MEDIA HELPERS
# ─────────────────────────────────────────────

def detect_media_type(upload) -> str:
    mimetype = (getattr(upload, 'mimetype', '') or '').lower()
    filename = (getattr(upload, 'filename', '') or '').lower()
    ext = os.path.splitext(filename)[1]
    if mimetype.startswith('video/') or ext in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}:
        return 'video'
    if mimetype.startswith('image/') or ext in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}:
        return 'image'
    if mimetype == 'text/plain' or ext == '.txt':
        return 'text'
    return 'unknown'


def extract_video_frames(video_path: str, job_id: str, max_frames: int = 4) -> list[str]:
    frame_dir = os.path.join(OUTPUT_FOLDER, f'{job_id}_frames')
    os.makedirs(frame_dir, exist_ok=True)
    subprocess.call([
        ffmpeg_cmd(), '-loglevel', 'error', '-y',
        '-i', video_path,
        '-vf', 'fps=0.25,scale=640:-1',
        '-frames:v', str(max_frames),
        os.path.join(frame_dir, 'frame_%02d.jpg'),
    ])
    frames = [
        os.path.join(frame_dir, f'frame_{i:02d}.jpg')
        for i in range(1, max_frames + 1)
        if os.path.exists(os.path.join(frame_dir, f'frame_{i:02d}.jpg'))
    ]
    return frames


def _vision_caption(content: list, api_key: str) -> str:
    client = _openai_client(api_key)
    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': content}],
        temperature=0.5,
        max_tokens=220,
    )
    return resp.choices[0].message.content.strip()


def describe_video_content(video_path: str, api_key: str, job_id: str) -> str:
    frame_files = []
    try:
        frame_files = extract_video_frames(video_path, job_id, max_frames=4)
        if not frame_files:
            return 'Shifting fragments, vivid but difficult to pin down.'
        content = [{
            'type': 'text',
            'text': (
                'These are still frames sampled from a short video. '
                'Describe what is visible and infer emotional/symbolic themes in 3-4 concise sentences.'
            ),
        }]
        for fp in frame_files:
            with open(fp, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{encoded}'}})
        return _vision_caption(content, api_key)
    except Exception as e:
        return f'I only caught fragments from the video memory: {e}'
    finally:
        for fp in frame_files:
            safe_remove(fp)
        frame_dir = os.path.join(OUTPUT_FOLDER, f'{job_id}_frames')
        if os.path.isdir(frame_dir):
            try:
                os.rmdir(frame_dir)
            except Exception:
                pass


def describe_image_content(image_path: str, api_key: str) -> str:
    try:
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        content = [
            {
                'type': 'text',
                'text': (
                    'This is a still image provided by the user. '
                    'Describe what is visible and infer emotional/symbolic themes in 3-4 concise sentences.'
                ),
            },
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{encoded}'}},
        ]
        return _vision_caption(content, api_key)
    except Exception as e:
        return f'I only caught fragments from the image memory: {e}'


def describe_media_content(media_path: str, media_type: str, api_key: str, job_id: str = '') -> str:
    if media_type == 'video':
        return describe_video_content(media_path, api_key, job_id or str(uuid.uuid4())[:8])
    if media_type == 'image':
        return describe_image_content(media_path, api_key)
    return 'Unfamiliar media fragment.'


# ─────────────────────────────────────────────
# IMAGE GENERATION
# ─────────────────────────────────────────────

def generate_image(prompt: str, api_key: str) -> str:
    try:
        client = _openai_client(api_key)
        resp = client.images.generate(
            model='dall-e-3',
            prompt=prompt,
            size='1024x1024',
            quality='hd',
            style='natural',
            n=1,
        )
        url = resp.data[0].url if getattr(resp, 'data', None) else None
        return url or 'Error: API returned no image URL.'
    except TypeError as e:
        return f'Error: OpenAI SDK parameter mismatch — {e}'
    except Exception as e:
        return f'Error: {e}'


# ─────────────────────────────────────────────
# DATAMOSH CORE
# ─────────────────────────────────────────────

def safe_remove(path: str):
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def schedule_delete(path: str, delay_seconds: int = 900):
    def worker():
        time.sleep(max(0, int(delay_seconds)))
        safe_remove(path)
    threading.Thread(target=worker, daemon=True).start()


def run_mosh(input_path: str, output_path: str, params: dict, job_id: str) -> dict:
    effect      = params.get('effect', 'iframe_removal')
    start_frame = int(params.get('start_frame', 0))
    end_frame   = int(params.get('end_frame', -1))
    fps         = max(1, min(120, int(params.get('fps', 30))))
    delta       = int(params.get('delta', 5)) if effect == 'delta_repeat' else 0

    input_avi  = f'tmp_{job_id}_in.avi'
    output_avi = f'tmp_{job_id}_out.avi'

    try:
        ret = subprocess.call([
            ffmpeg_cmd(), '-loglevel', 'error', '-y',
            '-i', input_path,
            '-crf', '0', '-pix_fmt', 'yuv420p', '-bf', '0',
            '-b', '10000k', '-r', str(fps),
            input_avi,
        ])
        if ret != 0 or not os.path.exists(input_avi):
            return {'success': False, 'error': 'ffmpeg AVI conversion failed.'}

        with open(input_avi, 'rb') as f:
            data = f.read()

        frame_marker = bytes.fromhex('30306463')
        iframe_sig   = bytes.fromhex('0001B0')
        pframe_sig   = bytes.fromhex('0001B6')

        frames = data.split(frame_marker)
        header = frames[0]
        frames = frames[1:]

        n_video_frames = sum(
            1 for fr in frames
            if len(fr) > 8 and (fr[5:8] == iframe_sig or fr[5:8] == pframe_sig)
        )
        if end_frame < 0:
            end_frame = n_video_frames

        with open(output_avi, 'wb') as out:
            out.write(header)
            if delta > 0:
                if delta > max(1, end_frame - start_frame):
                    return {
                        'success': False,
                        'error': f'Delta ({delta}) exceeds frame range ({end_frame - start_frame}).',
                    }
                repeat_frames = []
                repeat_index  = 0
                for idx, frame in enumerate(frames):
                    is_video = len(frame) > 8 and (frame[5:8] == iframe_sig or frame[5:8] == pframe_sig)
                    in_range = start_frame <= idx < end_frame
                    if not is_video or not in_range:
                        out.write(frame_marker + frame)
                        continue
                    if len(repeat_frames) < delta and frame[5:8] != iframe_sig:
                        repeat_frames.append(frame)
                        out.write(frame_marker + frame)
                    elif len(repeat_frames) == delta:
                        out.write(frame_marker + repeat_frames[repeat_index])
                        repeat_index = (repeat_index + 1) % delta
                    else:
                        out.write(frame_marker + frame)
            else:
                for idx, frame in enumerate(frames):
                    skip = (
                        len(frame) > 8
                        and frame[5:8] == iframe_sig
                        and start_frame <= idx <= end_frame
                    )
                    if not skip:
                        out.write(frame_marker + frame)

        ret = subprocess.call([
            ffmpeg_cmd(), '-loglevel', 'error', '-y',
            '-i', output_avi,
            '-crf', '18', '-pix_fmt', 'yuv420p',
            '-vcodec', 'libx264', '-acodec', 'aac',
            '-b', '10000k', '-r', str(fps),
            output_path,
        ])

        if os.path.exists(output_path):
            return {'success': True}
        return {'success': False, 'error': 'Output file not created.'}

    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        for tmp in [input_avi, output_avi]:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass


# ─────────────────────────────────────────────
# DREAM ENGINE
# ─────────────────────────────────────────────

dream_memories: dict = {}   # session_id -> list[dict]
dream_jobs:     dict = {}   # dream_id   -> dict

MEMORY_STORE_DIR = 'memory_store'
os.makedirs(MEMORY_STORE_DIR, exist_ok=True)


def _memory_path(session_id: str) -> str:
    return os.path.join(MEMORY_STORE_DIR, f'{sanitize_id(session_id)}.json')


def _save_memories(session_id: str):
    try:
        with open(_memory_path(session_id), 'w') as f:
            json.dump(dream_memories.get(session_id, []), f)
    except Exception as e:
        print(f'Memory save failed: {e}')


def _load_memories(session_id: str) -> list:
    path = _memory_path(session_id)
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def _ensure_memories(session_id: str) -> list:
    if session_id not in dream_memories:
        dream_memories[session_id] = _load_memories(session_id)
    return dream_memories[session_id]


def _append_text_memory(session_id: str, text: str) -> dict:
    memories = _ensure_memories(session_id)
    memory = {
        'memory_id': str(uuid.uuid4())[:8],
        'type':      'text',
        'caption':   text,
        'timestamp': time.time(),
    }
    memories.append(memory)
    _save_memories(session_id)
    return memory


def sanitize_id(s: str) -> str:
    return ''.join(c for c in s if c.isalnum() or c in {'-', '_'})


# ── Tone config ──────────────────────────────

DREAM_TONES = {
    'calm': {
        'style':        'soft diffused light, morning mist, wide open spaces, watercolor, slow motion',
        'narrative_mod': 'peaceful, drifting, weightless, gentle transitions',
        'tts_voice':    'nova',
        'tts_speed':    0.85,
        'scene_count':  3,
    },
    'surreal': {
        'style':        'impossible geometry, chromatic aberration, double exposure, melting edges, otherworldly color',
        'narrative_mod': 'non-sequitur logic, scale violations, entity merging, physics ignored',
        'tts_voice':    'onyx',
        'tts_speed':    0.95,
        'scene_count':  4,
    },
    'horror': {
        'style':        'deep shadow, flickering red light, desaturated, heavy film grain, claustrophobic framing',
        'narrative_mod': 'dread, repeating loops, wrong geometry, incomplete sentences',
        'tts_voice':    'shimmer',
        'tts_speed':    0.78,
        'scene_count':  4,
    },
    'nostalgic': {
        'style':        'faded film grain, warm amber tones, lomography, soft focus, overexposed edges',
        'narrative_mod': 'childhood logic, sensory memory, recursive self-reference, incomplete loops',
        'tts_voice':    'alloy',
        'tts_speed':    0.88,
        'scene_count':  3,
    },
}

SCENE_VARIATIONS = [
    'establishing shot, wide angle, dreamlike atmosphere',
    'close-up detail, texture and surface emphasis',
    'slight camera drift, motion blur at edges, temporal echo',
    'overexposed center, deep vignette, double exposure ghost',
]

TRANSITION_MOSH = {
    'cut':     {'effect': 'iframe_removal', 'start_frame': 0, 'end_frame': -1, 'delta': 0},
    'dissolve':{'effect': 'delta_repeat',   'start_frame': 2, 'end_frame': -1, 'delta': 2},
    'glitch':  {'effect': 'iframe_removal', 'start_frame': 1, 'end_frame': -1, 'delta': 0},
    'smear':   {'effect': 'delta_repeat',   'start_frame': 2, 'end_frame': -1, 'delta': 7},
    'freeze':  {'effect': 'iframe_removal', 'start_frame': 3, 'end_frame': 6,  'delta': 0},
}

_TRANSITION_TYPES = '|'.join(TRANSITION_MOSH.keys())

DREAM_NARRATIVE_SYSTEM = (
    'You are an AI generating a dream sequence as raw JSON only.\n'
    'Dreams are non-linear and illogical but emotionally coherent.\n'
    'Each scene must:\n'
    '- Begin mid-action, already happening (no "I was...")\n'
    '- Include at least one impossible or physically wrong element\n'
    '- End unresolved or bleeding into the next scene\n'
    '- Reference a fragment from today\'s memories\n'
    'Output ONLY valid JSON with this exact shape:\n'
    '{"scenes":[{"scene_id":0,"narration":"...","image_prompt":"...","dominant_emotion":"...",'
    f'"transition_type":"{_TRANSITION_TYPES}","duration_hint":5}}]}}'
)

DREAM_RECALL_SYSTEM = (
    'You just had a dream. You are recalling it upon waking — some parts are clear, others slip away.\n'
    'Rules:\n'
    '- First person, present-tense recall ("I remember... it felt like...")\n'
    '- Include a moment where you lose the thread ("and then... I can\'t remember")\n'
    '- Mention one detail with intense clarity and one with total uncertainty\n'
    '- End with an emotional residue, not a conclusion\n'
    '- Never be analytical. Never explain the dream.\n'
    '- Minimum 120 words. Maximum 220 words. The text will be read aloud (30s–1m30s of audio).'
)


# ── Dream helpers ─────────────────────────────

def _dream_update(dream_id: str, progress: int, message: str):
    dream_jobs[dream_id].update({'progress': progress, 'message': message})
    print(f'[Dream {dream_id}] {progress}% — {message}')


def infer_tone_from_story(bedtime_story: str, api_key: str) -> str:
    tones = ', '.join(DREAM_TONES.keys())
    try:
        resp = _openai_client(api_key).chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system', 'content': (
                    f'Classify the emotional atmosphere of this text into exactly one word. '
                    f'Choose from: {tones}. Return only that single word, nothing else.'
                )},
                {'role': 'user', 'content': bedtime_story},
            ],
            temperature=0.2,
            max_tokens=5,
        )
        tone = resp.choices[0].message.content.strip().lower()
        return tone if tone in DREAM_TONES else 'surreal'
    except Exception:
        return 'surreal'


def get_audio_duration(audio_path: str) -> float:
    try:
        r = subprocess.run(
            [ffprobe_cmd(), '-v', 'quiet', '-print_format', 'json', '-show_format', audio_path],
            capture_output=True, text=True, timeout=15,
        )
        return float(json.loads(r.stdout)['format']['duration'])
    except Exception:
        return 45.0


def ingest_memory(session_id: str, media_path: str, media_type: str, api_key: str) -> dict:
    caption = describe_media_content(media_path, media_type, api_key,
                                     job_id=f'{session_id}_{uuid.uuid4().hex[:4]}')
    memory = {
        'memory_id': str(uuid.uuid4())[:8],
        'type':      media_type,
        'caption':   caption,
        'timestamp': time.time(),
    }
    _ensure_memories(session_id).append(memory)
    _save_memories(session_id)
    return memory


def generate_dream_narrative(memories: list, bedtime_story: str,
                              tone: str, tone_config: dict, api_key: str) -> list:
    memory_text = '\n'.join(f'- {m["caption"]}' for m in memories) or '(no memories today)'
    user_msg = (
        f'Tone: {tone} — {tone_config["narrative_mod"]}\n'
        f'Bedtime story: {bedtime_story}\n'
        f'Today\'s memories:\n{memory_text}\n\n'
        f'Generate exactly {tone_config["scene_count"]} scenes.'
    )
    resp = _openai_client(api_key).chat.completions.create(
        model='gpt-4o',
        messages=[
            {'role': 'system', 'content': DREAM_NARRATIVE_SYSTEM},
            {'role': 'user',   'content': user_msg},
        ],
        response_format={'type': 'json_object'},
        temperature=1.1,
        max_tokens=1800,
    )
    return json.loads(resp.choices[0].message.content).get('scenes', [])


def generate_recall_text(scenes: list, bedtime_story: str, api_key: str) -> str:
    narrations = '\n'.join(f'Scene {s["scene_id"]}: {s["narration"]}' for s in scenes)
    resp = _openai_client(api_key).chat.completions.create(
        model='gpt-4o',
        messages=[
            {'role': 'system', 'content': DREAM_RECALL_SYSTEM},
            {'role': 'user',   'content': f'Dream scenes:\n{narrations}\n\nRecall this dream as if half-remembering.'},
        ],
        temperature=0.9,
        max_tokens=350,
    )
    return resp.choices[0].message.content.strip()


def generate_dream_tts(text: str, voice: str, speed: float, dream_id: str, api_key: str) -> str:
    audio_path = os.path.join(OUTPUT_FOLDER, f'{dream_id}_recall.mp3')
    response = _openai_client(api_key).audio.speech.create(
        model='tts-1-hd', voice=voice, input=text, speed=speed,
    )
    with open(audio_path, 'wb') as f:
        f.write(response.content)
    return audio_path


def build_scene_image_prompt(scene: dict, tone_config: dict) -> str:
    return f"{scene['image_prompt']}, {tone_config['style']}, cinematic, photorealistic dream, 8K"


def make_dream_clip(image_path: str, duration: int, dream_id: str,
                    clip_id: str, transition_type: str = 'smear') -> str | None:
    src_path  = os.path.join(OUTPUT_FOLDER, f'{dream_id}_{clip_id}_src.mp4')
    out_path  = os.path.join(OUTPUT_FOLDER, f'{dream_id}_{clip_id}.mp4')
    fps       = 12
    total_dur = max(2, duration)
    n_frames  = total_dur * fps

    ret = subprocess.call([
        ffmpeg_cmd(), '-loglevel', 'error', '-y',
        '-loop', '1', '-i', image_path,
        '-t', str(total_dur),
        '-vf', (
            'scale=1024:1024:force_original_aspect_ratio=decrease,'
            'pad=1024:1024:(ow-iw)/2:(oh-ih)/2,'
            f"zoompan=z='min(zoom+0.0015,1.1)':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f'd={n_frames}:s=1024x1024,'
            f'fps={fps},format=yuv420p'
        ),
        '-r', str(fps), '-pix_fmt', 'yuv420p', '-an',
        src_path,
    ])
    if ret != 0 or not os.path.exists(src_path):
        return None

    mosh_params = {**TRANSITION_MOSH.get(transition_type, TRANSITION_MOSH['smear']), 'fps': fps}
    result = run_mosh(src_path, out_path, mosh_params, f'{dream_id}_{clip_id}_mosh')
    safe_remove(src_path)
    return out_path if result.get('success') and os.path.exists(out_path) else None


def assemble_dream_video(clips: list, audio_path: str | None, dream_id: str) -> str | None:
    silent_path = os.path.join(OUTPUT_FOLDER, f'{dream_id}_silent.mp4')
    final_path  = os.path.join(OUTPUT_FOLDER, f'{dream_id}_dream.mp4')
    concat_file = os.path.join(OUTPUT_FOLDER, f'{dream_id}_cc.txt')

    with open(concat_file, 'w') as f:
        for c in clips:
            f.write(f"file '{os.path.abspath(c)}'\n")

    ret = subprocess.call([
        ffmpeg_cmd(), '-loglevel', 'error', '-y',
        '-f', 'concat', '-safe', '0', '-i', concat_file,
        '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-r', '12',
        silent_path,
    ])
    safe_remove(concat_file)

    if ret != 0 or not os.path.exists(silent_path):
        return None

    if audio_path and os.path.exists(audio_path):
        ret = subprocess.call([
            ffmpeg_cmd(), '-loglevel', 'error', '-y',
            '-i', silent_path, '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'aac', '-shortest',
            final_path,
        ])
        safe_remove(silent_path)
        return final_path if ret == 0 and os.path.exists(final_path) else None

    os.rename(silent_path, final_path)
    return final_path


# ── Pipeline ──────────────────────────────────

def _download_image(url: str, path: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            with open(path, 'wb') as fh:
                fh.write(resp.read())
        return True
    except Exception as e:
        print(f'Image download failed: {e}')
        return False


def _new_dream_id(session_id: str) -> str:
    # Evict completed/error jobs beyond 50
    done = [k for k, v in dream_jobs.items() if v.get('status') in {'complete', 'error', 'recall_complete'}]
    if len(done) > 50:
        for k in done[:len(done) - 50]:
            dream_jobs.pop(k, None)

    dream_id = str(uuid.uuid4())[:8]
    dream_jobs[dream_id] = {
        'status': 'running', 'progress': 0, 'message': 'Starting…',
        'session_id': session_id, 'created_at': time.time(),
    }
    return dream_id


def run_dream_recall_pipeline(dream_id: str, session_id: str,
                               bedtime_story: str, api_key: str):
    """Phase A: infer tone → narrative → recall text."""
    try:
        memories = _ensure_memories(session_id)

        _dream_update(dream_id, 5,  'Entering dream state…')
        _dream_update(dream_id, 10, 'Reading the story…')
        tone        = infer_tone_from_story(bedtime_story, api_key)
        tone_config = DREAM_TONES[tone]

        _dream_update(dream_id, 20, 'Generating dream narrative…')
        scenes = generate_dream_narrative(memories, bedtime_story, tone, tone_config, api_key)
        if not scenes:
            raise ValueError('Narrative engine returned no scenes.')

        _dream_update(dream_id, 70, 'Waking from dream…')
        recall_text = generate_recall_text(scenes, bedtime_story, api_key)

        dream_jobs[dream_id].update({
            'status':      'recall_complete',
            'progress':    100,
            'message':     'Dream recall ready.',
            'scenes':      scenes,
            'tone':        tone,
            'tone_config': tone_config,
            'recall': {
                'recall_text':   recall_text,
                'inferred_tone': tone,
            },
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        dream_jobs[dream_id].update({
            'status': 'error', 'progress': 0,
            'message': f'Dream failed: {exc}', 'error': str(exc),
        })


def run_dream_visualize_pipeline(dream_id: str, image_count: int, api_key: str):
    """Phase B: generate N images → datamosh → TTS → assemble video."""
    try:
        job         = dream_jobs[dream_id]
        scenes      = job['scenes']
        tone_config = job['tone_config']
        tone        = job['tone']
        image_count = max(1, min(6, image_count))

        dream_jobs[dream_id].update({'status': 'running', 'progress': 5, 'message': 'Generating images…'})

        # Build prompts
        prompts = [
            (k, scenes[k % len(scenes)],
             f'{build_scene_image_prompt(scenes[k % len(scenes)], tone_config)}, {SCENE_VARIATIONS[k % len(SCENE_VARIATIONS)]}')
            for k in range(image_count)
        ]

        # Generate images in parallel
        image_paths: list[str | None] = [None] * image_count

        def _gen(args):
            k, scene, prompt = args
            url = generate_image(prompt, api_key)
            if url.startswith('Error'):
                return k, None
            path = os.path.join(OUTPUT_FOLDER, f'{dream_id}_vis{k}.jpg')
            return k, path if _download_image(url, path) else None

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_gen, p): p[0] for p in prompts}
            done = 0
            for fut in as_completed(futures):
                k, path = fut.result()
                image_paths[k] = path
                done += 1
                _dream_update(dream_id, 5 + int((done / image_count) * 55), f'Generated {done}/{image_count} images…')

        if not any(image_paths):
            raise ValueError('No images could be generated.')

        # TTS
        _dream_update(dream_id, 63, 'Synthesising voice…')
        recall_text = job.get('recall', {}).get('recall_text', '')
        audio_path  = None
        if recall_text:
            try:
                audio_path = generate_dream_tts(
                    recall_text, tone_config['tts_voice'], tone_config['tts_speed'], dream_id, api_key,
                )
            except Exception as e:
                print(f'TTS failed: {e}')

        # Clip duration matched to audio length
        total_dur = get_audio_duration(audio_path) if audio_path else image_count * 5.0
        per_clip  = max(2, int(total_dur / max(1, image_count)))

        # Datamosh each image → clip
        _dream_update(dream_id, 70, 'Applying datamosh…')
        clips = []
        for k, img_path in enumerate(image_paths):
            if not img_path:
                continue
            transition_type = scenes[k % len(scenes)].get('transition_type', 'smear')
            clip = make_dream_clip(img_path, per_clip, dream_id, f'vis{k}', transition_type)
            if clip:
                clips.append(clip)

        if not clips:
            raise ValueError('Datamosh step produced no video clips.')

        # Assemble
        _dream_update(dream_id, 88, 'Assembling dream video…')
        final_video = assemble_dream_video(clips, audio_path, dream_id)
        for c in clips:
            safe_remove(c)

        # Collect image URLs and schedule cleanup
        image_urls = []
        for p in image_paths:
            if p and os.path.exists(p):
                image_urls.append(f'/api/dream-image/{os.path.basename(p)}')
                schedule_delete(p, delay_seconds=7200)
        if final_video:
            schedule_delete(final_video, delay_seconds=7200)
        if audio_path:
            schedule_delete(audio_path, delay_seconds=7200)

        dream_jobs[dream_id].update({
            'status': 'complete', 'progress': 100, 'message': 'Dream complete.',
            'result': {
                'recall_text':   recall_text,
                'image_urls':    image_urls,
                'audio_url':     (f'/api/dream-audio/{dream_id}'
                                  if audio_path and os.path.exists(audio_path) else None),
                'video_url':     f'/api/dream-video/{dream_id}' if final_video else None,
                'inferred_tone': tone,
            },
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        dream_jobs[dream_id].update({
            'status': 'error', 'progress': 0,
            'message': f'Visualization failed: {exc}', 'error': str(exc),
        })


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('dream.html')


@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('videos', filename)


@app.route('/api/ping')
def ping():
    ffmpeg_path = ffmpeg_cmd()
    try:
        ffmpeg_ok = subprocess.call(
            [ffmpeg_path, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ) == 0
    except Exception:
        ffmpeg_ok = False
    return jsonify({
        'ok': True,
        'ffmpeg': ffmpeg_ok,
        'ffmpeg_path': ffmpeg_path,
        'hint': ffmpeg_install_hint(),
        'openai_configured': bool(get_server_api_key()),
    })


# ── Dream memory routes ───────────────────────

@app.route('/api/dream/ingest', methods=['POST'])
def dream_ingest():
    session_id = sanitize_id(request.form.get('session_id', '').strip()) or str(uuid.uuid4())[:8]
    api_key    = get_server_api_key()
    if not api_key:
        return jsonify({'error': 'OpenAI API key required'}), 400

    ingested = []

    for upload in request.files.getlist('media'):
        if not upload or not upload.filename:
            continue
        media_type = detect_media_type(upload)
        if media_type == 'unknown':
            continue
        if media_type == 'text':
            try:
                content = upload.read().decode('utf-8', errors='replace').strip()
                if content:
                    ingested.append(_append_text_memory(session_id, content[:4000]))
            except Exception:
                pass
            continue
        ext      = os.path.splitext(upload.filename)[1] or ('.mp4' if media_type == 'video' else '.jpg')
        tmp_path = os.path.join(UPLOAD_FOLDER, f'{session_id}_{uuid.uuid4().hex[:6]}_mem{ext}')
        upload.save(tmp_path)
        ingested.append(ingest_memory(session_id, tmp_path, media_type, api_key))
        safe_remove(tmp_path)

    text_input = request.form.get('text', '').strip()
    if text_input:
        ingested.append(_append_text_memory(session_id, text_input))

    memories = dream_memories.get(session_id, [])
    return jsonify({
        'session_id':     session_id,
        'ingested':       len(ingested),
        'total_memories': len(memories),
        'memories': [{'id': m['memory_id'], 'type': m['type'], 'caption': m['caption'][:120]}
                     for m in memories],
    })


@app.route('/api/dream/memory/delete', methods=['POST'])
def dream_delete_memory():
    data       = request.get_json() or {}
    session_id = sanitize_id(data.get('session_id', ''))
    memory_id  = data.get('memory_id', '')
    memories   = _ensure_memories(session_id)
    dream_memories[session_id] = [m for m in memories if m['memory_id'] != memory_id]
    _save_memories(session_id)
    return jsonify({
        'ok': True,
        'memories': [{'id': m['memory_id'], 'type': m['type'], 'caption': m['caption']}
                     for m in dream_memories[session_id]],
    })


@app.route('/api/dream/memories/<session_id>')
def dream_get_memories(session_id):
    session_id = sanitize_id(session_id)
    memories   = _ensure_memories(session_id)
    return jsonify({
        'session_id': session_id,
        'memories': [{'id': m['memory_id'], 'type': m['type'], 'caption': m['caption']}
                     for m in memories],
    })


@app.route('/api/dream/clear/<session_id>', methods=['POST'])
def dream_clear_memories(session_id):
    session_id = sanitize_id(session_id)
    dream_memories.pop(session_id, None)
    safe_remove(_memory_path(session_id))
    return jsonify({'ok': True})


# ── Dream pipeline routes ─────────────────────

@app.route('/api/dream/recall', methods=['POST'])
def dream_recall_route():
    data          = request.get_json() or {}
    session_id    = sanitize_id(data.get('session_id', '').strip())
    bedtime_story = data.get('bedtime_story', '').strip()

    if not session_id or not bedtime_story:
        return jsonify({'error': 'session_id and bedtime_story are required'}), 400

    api_key = get_server_api_key()
    if not api_key:
        return jsonify({'error': 'OpenAI API key required'}), 400

    dream_id = _new_dream_id(session_id)
    threading.Thread(
        target=run_dream_recall_pipeline,
        args=(dream_id, session_id, bedtime_story, api_key),
        daemon=True,
    ).start()
    return jsonify({'dream_id': dream_id})


@app.route('/api/dream/visualize', methods=['POST'])
def dream_visualize_route():
    data        = request.get_json() or {}
    dream_id    = sanitize_id(data.get('dream_id', '').strip())
    image_count = max(1, min(6, int(data.get('image_count', 3))))

    job = dream_jobs.get(dream_id)
    if not job:
        return jsonify({'error': 'Dream not found'}), 404
    if job.get('status') != 'recall_complete':
        return jsonify({'error': 'Phase A not complete yet'}), 400

    api_key = get_server_api_key()
    if not api_key:
        return jsonify({'error': 'OpenAI API key required'}), 400

    dream_jobs[dream_id].update({'status': 'running', 'progress': 0, 'message': 'Starting…'})
    threading.Thread(
        target=run_dream_visualize_pipeline,
        args=(dream_id, image_count, api_key),
        daemon=True,
    ).start()
    return jsonify({'dream_id': dream_id})


@app.route('/api/dream/status/<dream_id>')
def dream_status(dream_id):
    dream_id = sanitize_id(dream_id)
    job = dream_jobs.get(dream_id)
    if not job:
        return jsonify({'error': 'Dream not found'}), 404
    resp = {
        'status':   job['status'],
        'progress': job['progress'],
        'message':  job['message'],
        'error':    job.get('error'),
    }
    if job['status'] == 'recall_complete':
        resp['recall'] = job.get('recall', {})
    return jsonify(resp)


@app.route('/api/dream/result/<dream_id>')
def dream_result(dream_id):
    dream_id = sanitize_id(dream_id)
    job = dream_jobs.get(dream_id)
    if not job:
        return jsonify({'error': 'Dream not found'}), 404
    if job['status'] != 'complete':
        return jsonify({'error': 'Not ready', 'status': job['status']}), 202
    return jsonify(job['result'])


# ── Dream media serve routes ──────────────────

@app.route('/api/dream-video/<dream_id>')
def serve_dream_video(dream_id):
    dream_id = sanitize_id(dream_id)
    try:
        return send_file(os.path.join(OUTPUT_FOLDER, f'{dream_id}_dream.mp4'), mimetype='video/mp4')
    except FileNotFoundError:
        return jsonify({'error': 'Dream video not found'}), 404


@app.route('/api/dream-audio/<dream_id>')
def serve_dream_audio(dream_id):
    dream_id = sanitize_id(dream_id)
    try:
        return send_file(os.path.join(OUTPUT_FOLDER, f'{dream_id}_recall.mp3'), mimetype='audio/mpeg')
    except FileNotFoundError:
        return jsonify({'error': 'Dream audio not found'}), 404


@app.route('/api/dream-image/<filename>')
def serve_dream_image(filename):
    safe_name = ''.join(c for c in filename if c.isalnum() or c in {'-', '_', '.'})
    try:
        return send_from_directory(OUTPUT_FOLDER, safe_name)
    except FileNotFoundError:
        return jsonify({'error': 'Image not found'}), 404


# ─────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────

@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({'error': 'Uploaded file is too large. Limit is 500 MB.'}), 413


@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Resource not found'}), 404


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print('\n=== AI Dream Engine ===')
    ffmpeg_path = ffmpeg_cmd()
    try:
        ok = subprocess.call([ffmpeg_path, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
        print(f'ffmpeg: {"found" if ok else "NOT FOUND"} ({ffmpeg_path})')
        if not ok:
            print(' ', ffmpeg_install_hint())
    except FileNotFoundError:
        print(f'ffmpeg not found. {ffmpeg_install_hint()}')

    host = '0.0.0.0'
    port = 5001
    if is_port_in_use('127.0.0.1', port) or is_port_in_use('0.0.0.0', port):
        print(f'Cannot start: port {port} is already in use.')
        sys.exit(1)

    print(f'Open http://127.0.0.1:{port}')
    app.run(debug=False, host=host, port=port)
