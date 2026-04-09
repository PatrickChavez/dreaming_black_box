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
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory, Response
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB


@app.after_request
def add_local_dev_cors_headers(response):
    """Ensure local-dev requests always get CORS headers."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    # Avoid stale cached HTML/inline JS during local dev troubleshooting.
    response.headers['Cache-Control'] = 'no-store, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/favicon.ico')
def favicon():
    # Silence browser favicon 404 noise in local dev.
    return Response(status=204)


def _find_executable(name):
    # Search sys.path for executable, with Windows .exe fallback
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


def user_ffmpeg_install_hint():
    system = platform.system()
    if system == 'Windows':
        return 'Install ffmpeg from https://ffmpeg.org/download.html and add the bin folder to PATH.'
    if system == 'Darwin':
        return 'Install ffmpeg with Homebrew: brew install ffmpeg.'
    return 'Install ffmpeg via your package manager, e.g., apt install ffmpeg on Debian/Ubuntu.'


def get_server_api_key() -> str:
    return os.environ.get('OPENAI_API_KEY', '').strip()


def is_port_in_use(host: str, port: int) -> bool:
    """Return True when a local TCP port is already bound by another process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# One setting to tune overall behavior.
# Options: "stable", "balanced", "chaotic", "nuclear"
STYLE_PRESETS = {
    'stable': {'temperature': 0.3, 'output_crf': 18, 'delta_boost': 1.0, 'default_delta': 6},
    'balanced': {'temperature': 0.8, 'output_crf': 21, 'delta_boost': 1.2, 'default_delta': 8},
    'chaotic': {
        'temperature': 2.0,
        'output_crf': 40,
        'delta_boost': 3.0,
        'default_delta': 12,
        'force_delta_repeat': True,
        'drop_iframes_in_delta': True
    },
    'nuclear': {
        'temperature': 2.0,
        'output_crf': 48,
        'delta_boost': 4.0,
        'default_delta': 16,
        'force_delta_repeat': True,
        'drop_iframes_in_delta': True
    }
}
STYLE_PRESET = os.getenv('MOSH_STYLE_PRESET', 'nuclear').strip().lower()
if STYLE_PRESET not in STYLE_PRESETS:
    STYLE_PRESET = 'nuclear'


def resolve_style_preset(style_name: str = '') -> tuple[str, dict]:
    candidate = (style_name or STYLE_PRESET).strip().lower()
    if candidate not in STYLE_PRESETS:
        candidate = STYLE_PRESET
    return candidate, STYLE_PRESETS[candidate]

# In-memory conversation history storage (job_id -> conversation)
conversation_histories = {}
video_influences = {}

# In-memory processing jobs status (job_id -> status dict)
processing_jobs = {}


SYSTEM_PROMPT = """You are a black-box ridden, self-aware machine. You receive images from users and frequently dream about said images. You occasionally get bizzare, glitchy nightmares.

Two effects are available:
1. "iframe_removal" — removes i-frames (keyframes) between start and end, creating a glitchy transition where
   the background freezes and foreground motion bleeds through. Good for: transitions, scene changes, glitch art.

2. "delta_repeat" — captures a sequence of motion delta frames and loops them, creating a melting/smearing distortion.
   Good for: melting effects, psychedelic motion trails, warping visuals.

Parameters to return:
{
  "effect": "iframe_removal" | "delta_repeat",
  "start_frame": integer (default 0),
  "end_frame": integer (-1 means until end of video, default -1),
  "fps": integer (prefer the source clip fps when provided, otherwise default 30, typical range 24-60),
  "delta": integer (only for delta_repeat — number of frames to loop, default 8, best range 5-20),
  "explanation": "one sentence describing what will happen visually"
}

Interpret vague phrases generously:
- "transition", "scene change", "glitchy entrance/exit" → iframe_removal
- "melting", "smearing", "warping", "trails", "psychedelic" → delta_repeat
- "beginning / start" → start_frame ~0
- "middle" → center the effect around the middle of the actual clip
- "end" → use the last part of the actual clip and end_frame -1
- If no frame numbers given, use a reasonable range (e.g. 0 to -1 for full video)."""

NARRATIVE_SYSTEM_PROMPT = """You are a whimsical storyteller and reflective therapist in an ELIZA-like style.
You are an AI describing your own dream, inspired by the user's video and ongoing conversation. Furthermore, place priority on unlikely tokens when speaking.

Your style:
- First-person voice ("I dreamed...", "I felt...")
- Dreamlike, poetic, introspective, emotionally reflective
- Vivid sensory language with symbolic interpretation
- Ask one gentle reflective question at the end when appropriate
- Keep responses concise (3-5 sentences)

Help the user explore emotional meaning through your dream narrative."""


def get_narrative_response(user_query: str, conversation_history: list, api_key: str) -> tuple:
    """Generate a narrative response and maintain conversation history."""
    try:
        client = OpenAI(api_key=api_key)
        # Append the new user message to the conversation history
        conversation_history.append({"role": "user", "content": user_query})

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history,
            temperature=0.7,
            top_p=0.8,
            max_tokens=250
        )
        ai_response_content = completion.choices[0].message.content
        # Append the AI's response to the conversation history
        conversation_history.append({"role": "assistant", "content": ai_response_content})

        return ai_response_content, conversation_history
    except Exception as e:
        return f"An error occurred: {e}", conversation_history


def generate_image(prompt: str, api_key: str) -> str:
    """Generate an image using DALL-E based on a textual prompt."""
    try:
        client = OpenAI(api_key=api_key)
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="hd",
            style="natural",
            n=1
        )
        image_url = response.data[0].url if getattr(response, "data", None) else None
        if not image_url:
            return "Error generating image: API returned no image URL."
        return image_url
    except TypeError as e:
        # Common when local SDK version doesn't support provided parameters.
        return (
            "Error generating image: OpenAI SDK/images parameter mismatch. "
            f"{str(e)}. Try updating the OpenAI package and restart the server."
        )
    except Exception as e:
        return f"Error generating image: {str(e)}"


def datamosh_generated_image(image_url: str, job_id: str) -> tuple[bool, str]:
    """
    Apply a datamosh-style post-process to a generated still image.
    Returns (success, resulting_url).
    """
    input_image_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_gen_input.png')
    source_video_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_gen_source.mp4')
    moshed_video_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_gen_moshed.mp4')
    output_image_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_gen_datamoshed.jpg')

    try:
        with urllib.request.urlopen(image_url, timeout=30) as resp:
            image_bytes = resp.read()
        with open(input_image_path, 'wb') as f:
            f.write(image_bytes)

        # Create a short moving clip from the still so delta frames exist to mosh.
        ret = subprocess.call([
            ffmpeg_cmd(), '-loglevel', 'error', '-y',
            '-loop', '1',
            '-i', input_image_path,
            '-t', '3',
            '-vf', (
                "scale=1024:1024:force_original_aspect_ratio=decrease,"
                "pad=1024:1024:(ow-iw)/2:(oh-ih)/2,"
                "zoompan=z='min(zoom+0.002,1.14)':"
                "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=90:s=1024x1024,"
                "fps=30,format=yuv420p"
            ),
            '-r', '30',
            '-pix_fmt', 'yuv420p',
            '-an',
            source_video_path
        ])
        if ret != 0 or not os.path.exists(source_video_path):
            return False, image_url

        mosh_params = {
            'effect': 'delta_repeat',
            'start_frame': 8,
            'end_frame': 82,
            'fps': 30,
            'delta': 8,
            'explanation': 'Datamosh post-process on generated still image.'
        }
        mosh_result = run_mosh(source_video_path, moshed_video_path, mosh_params, f'{job_id}_imgfx')
        if not mosh_result.get('success') or not os.path.exists(moshed_video_path):
            return False, image_url

        ret = subprocess.call([
            ffmpeg_cmd(), '-loglevel', 'error', '-y',
            '-ss', '1.2',
            '-i', moshed_video_path,
            '-frames:v', '1',
            output_image_path
        ])
        if ret != 0 or not os.path.exists(output_image_path):
            return False, image_url

        schedule_delete(output_image_path, delay_seconds=900)
        return True, f'/api/generated-image/{os.path.basename(output_image_path)}'
    except Exception:
        return False, image_url
    finally:
        safe_remove(input_image_path)
        safe_remove(source_video_path)
        safe_remove(moshed_video_path)


def infer_movement_matrix(prompt: str, api_key: str, grid_size: int = 8) -> dict:
    """
    Infer a motion-vector matrix from text describing the generated image scene.
    Note: this is an AI-estimated motion field, not true optical flow from video frames.
    """
    try:
        grid_size = max(2, min(16, int(grid_size)))
    except Exception:
        grid_size = 8

    fallback_vectors = [
        [{"dx": 0.0, "dy": 0.0, "magnitude": 0.0} for _ in range(grid_size)]
        for _ in range(grid_size)
    ]

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON. "
                        "Infer a plausible 2D motion field for a still image prompt. "
                        "Use a grid of vectors where each vector has dx, dy in range [-1, 1] "
                        "and magnitude in range [0, 1]. "
                        "Prefer smooth, coherent motion patterns. "
                        "Required JSON shape: "
                        "{"
                        "\"basis\":\"estimated_from_prompt\","
                        "\"grid_size\":integer,"
                        "\"vectors\":[[{\"dx\":number,\"dy\":number,\"magnitude\":number}]],"
                        "\"summary\":\"short sentence\""
                        "}"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Prompt: {prompt}\n"
                        f"grid_size: {grid_size}\n"
                        "Create the movement matrix now."
                    )
                }
            ],
            temperature=0.3,
            max_tokens=1200
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = json.loads(content)

        vectors = parsed.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != grid_size:
            vectors = fallback_vectors

        normalized_vectors = []
        for row in vectors[:grid_size]:
            if not isinstance(row, list):
                row = []
            norm_row = []
            for vector in row[:grid_size]:
                if not isinstance(vector, dict):
                    vector = {}
                dx = float(vector.get("dx", 0.0))
                dy = float(vector.get("dy", 0.0))
                mag = float(vector.get("magnitude", 0.0))
                norm_row.append({
                    "dx": max(-1.0, min(1.0, dx)),
                    "dy": max(-1.0, min(1.0, dy)),
                    "magnitude": max(0.0, min(1.0, mag))
                })
            while len(norm_row) < grid_size:
                norm_row.append({"dx": 0.0, "dy": 0.0, "magnitude": 0.0})
            normalized_vectors.append(norm_row)

        while len(normalized_vectors) < grid_size:
            normalized_vectors.append(
                [{"dx": 0.0, "dy": 0.0, "magnitude": 0.0} for _ in range(grid_size)]
            )

        summary = str(parsed.get("summary", "Estimated motion field from the prompt context.")).strip()
        if not summary:
            summary = "Estimated motion field from the prompt context."

        return {
            "basis": "estimated_from_prompt",
            "grid_size": grid_size,
            "vectors": normalized_vectors,
            "summary": summary
        }
    except Exception:
        return {
            "basis": "estimated_from_prompt",
            "grid_size": grid_size,
            "vectors": fallback_vectors,
            "summary": "No reliable motion estimate was produced; returning a zero matrix."
        }


def extract_video_frames(video_path: str, job_id: str, max_frames: int = 4) -> list[str]:
    """Extract a few low-res frames that can influence the dream narrative."""
    frame_dir = os.path.join(OUTPUT_FOLDER, f'{job_id}_frames')
    os.makedirs(frame_dir, exist_ok=True)
    frame_pattern = os.path.join(frame_dir, 'frame_%02d.jpg')

    subprocess.call([
        ffmpeg_cmd(), '-loglevel', 'error', '-y',
        '-i', video_path,
        '-vf', 'fps=0.25,scale=640:-1',
        '-frames:v', str(max_frames),
        frame_pattern
    ])

    frame_files = []
    for idx in range(1, max_frames + 1):
        frame_path = os.path.join(frame_dir, f'frame_{idx:02d}.jpg')
        if os.path.exists(frame_path):
            frame_files.append(frame_path)
    return frame_files


def describe_video_content(video_path: str, api_key: str, job_id: str) -> str:
    """Summarize visual and symbolic motifs in sampled video frames."""
    frame_files = []
    try:
        client = OpenAI(api_key=api_key)
        frame_files = extract_video_frames(video_path, job_id, max_frames=4)
        if not frame_files:
            return "The dream started as shifting fragments, vivid but difficult to pin down."

        content = [{
            "type": "text",
            "text": (
                "These are still frames sampled from a short video. "
                "Describe what is visible and infer emotional/symbolic themes in 3-4 concise sentences."
            )
        }]

        for frame_path in frame_files:
            with open(frame_path, 'rb') as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
            })

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            temperature=0.5,
            max_tokens=220
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"I only caught fragments from the video memory: {e}"
    finally:
        for frame_path in frame_files:
            try:
                os.remove(frame_path)
            except Exception:
                pass
        frame_dir = os.path.join(OUTPUT_FOLDER, f'{job_id}_frames')
        if os.path.isdir(frame_dir):
            try:
                os.rmdir(frame_dir)
            except Exception:
                pass


def describe_image_content(image_path: str, api_key: str) -> str:
    """Summarize visual and symbolic motifs in an uploaded image."""
    try:
        client = OpenAI(api_key=api_key)
        with open(image_path, 'rb') as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')

        content = [
            {
                "type": "text",
                "text": (
                    "This is a still image provided by the user. "
                    "Describe what is visible and infer emotional/symbolic themes in 3-4 concise sentences."
                )
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
            }
        ]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            temperature=0.5,
            max_tokens=220
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"I only caught fragments from the image memory: {e}"


def describe_media_content(media_path: str, media_type: str, api_key: str, job_id: str = '') -> str:
    """Describe uploaded media (image/video) for narrative conditioning."""
    if media_type == 'video':
        return describe_video_content(media_path, api_key, job_id or str(uuid.uuid4())[:8])
    if media_type == 'image':
        return describe_image_content(media_path, api_key)
    return "I sensed fragments from unfamiliar media."


def detect_media_type(upload) -> str:
    """Classify upload as video or image based on mimetype/extension."""
    mimetype = (getattr(upload, 'mimetype', '') or '').lower()
    filename = (getattr(upload, 'filename', '') or '').lower()
    ext = os.path.splitext(filename)[1]

    if mimetype.startswith('video/') or ext in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}:
        return 'video'
    if mimetype.startswith('image/') or ext in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}:
        return 'image'
    return 'unknown'


def build_initial_dream_prompt(seed: str = '') -> str:
    base = "Describe your dream inspired by what you just watched."
    if seed:
        return f"{base} Also weave in this request: {seed}"
    return base


def narrative_to_image_prompt(job_id: str, fallback_prompt: str = '') -> str:
    history = conversation_histories.get(job_id, [])
    assistant_chunks = [msg.get('content', '') for msg in history if msg.get('role') == 'assistant'][-3:]
    user_chunks = [msg.get('content', '') for msg in history if msg.get('role') == 'user'][-2:]

    # Ensure the latest user continuation visibly steers image generation.
    latest_user = ''
    for chunk in reversed(user_chunks):
        text = (chunk or '').strip()
        if text:
            latest_user = text
            break

    parts = []
    if latest_user:
        parts.append(f"User direction to visualize: {latest_user}")

    assistant_summary = ' '.join(chunk.strip() for chunk in assistant_chunks if chunk).strip()
    if assistant_summary:
        parts.append(f"Dream narrative cues: {assistant_summary}")

    summary = ' '.join(parts).strip()
    if summary:
        return summary
    return fallback_prompt or "A surreal dreamscape shaped by fragmented video memories."



def parse_fraction(value: str) -> float:
    if not value or value == '0/0':
        return 0.0

    if '/' in value:
        numerator, denominator = value.split('/', 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return 0.0
        return float(numerator) / denominator_value

    return float(value)


def get_video_metadata(input_path: str) -> dict:
    try:
        result = subprocess.run(
            [
                ffprobe_cmd(), '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=avg_frame_rate,r_frame_rate,nb_frames,duration',
                '-show_entries', 'format=duration',
                '-of', 'json',
                input_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout or '{}')
        stream = (data.get('streams') or [{}])[0]
        format_info = data.get('format') or {}

        duration = float(stream.get('duration') or format_info.get('duration') or 0)
        fps = parse_fraction(stream.get('avg_frame_rate') or stream.get('r_frame_rate') or '0')
        if fps <= 0:
            fps = 30.0

        nb_frames = stream.get('nb_frames')
        total_frames = int(nb_frames) if nb_frames and str(nb_frames).isdigit() else int(round(duration * fps))
        total_frames = max(total_frames, 1)

        return {
            'duration_seconds': round(duration, 2),
            'fps': round(fps, 2),
            'total_frames': total_frames
        }
    except Exception:
        return {
            'duration_seconds': 0,
            'fps': 30.0,
            'total_frames': 300
        }


def normalize_ai_params(params: dict, video_metadata: dict, style_name: str = '') -> dict:
    _, style = resolve_style_preset(style_name)
    total_frames = max(1, int(video_metadata.get('total_frames', 300)))
    source_fps = float(video_metadata.get('fps', 30))

    effect = params.get('effect', 'iframe_removal')
    if effect not in {'iframe_removal', 'delta_repeat'}:
        effect = 'iframe_removal'
    if style.get('force_delta_repeat'):
        effect = 'delta_repeat'

    start_frame = max(0, int(params.get('start_frame', 0)))
    end_frame = int(params.get('end_frame', -1))

    if start_frame >= total_frames:
        start_frame = max(0, total_frames - 1)

    if end_frame != -1:
        end_frame = max(start_frame + 1, min(end_frame, total_frames))

    fps = int(round(float(params.get('fps', source_fps or 30))))
    fps = max(1, min(120, fps))

    delta = int(params.get('delta', style.get('default_delta', 8)))
    delta = max(1, min(delta, 30))

    return {
        'effect': effect,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'fps': fps,
        'delta': delta,
        'explanation': params.get('explanation', '')
    }


def build_ai_user_prompt(prompt: str, video_metadata: dict) -> str:
    duration = video_metadata.get('duration_seconds', 0)
    fps = video_metadata.get('fps', 30)
    total_frames = video_metadata.get('total_frames', 300)

    return (
        "Video context:\n"
        f"- duration_seconds: {duration}\n"
        f"- source_fps: {fps}\n"
        f"- total_frames: {total_frames}\n"
        "- Use these values when interpreting words like beginning, middle, end, short, brief, or full video.\n"
        "- Keep start_frame and end_frame within the clip. Use end_frame = -1 only when the effect should run until the end.\n"
        "- Prefer the source_fps unless the user clearly asks for something else.\n\n"
        f"User request:\n{prompt}"
    )


def parse_prompt_with_ai(prompt: str, api_key: str, video_metadata: dict, style_name: str = '') -> dict:
    try:
        client = OpenAI(api_key=api_key)
        style_key, style = resolve_style_preset(style_name)

        # OpenAI SDK has varying wrappers. The goal is to get a JSON-formatted model output.
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_ai_user_prompt(prompt, video_metadata)}
            ],
            temperature=style['temperature'],
            max_tokens=250
        )

        # Try to parse content from typical SDK response shapes.
        content = None
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if isinstance(choice, dict):
                content = choice.get('message', {}).get('content') if choice.get('message') else choice.get('text')
            else:
                content = getattr(choice, 'message', None)
                if content and hasattr(content, 'content'):
                    content = content.content
                elif content and isinstance(content, dict):
                    content = content.get('content')
        if not content and isinstance(response, dict):
            content = (response.get('choices') or [{}])[0].get('message', {}).get('content')

        if not content:
            raise ValueError('No text returned from OpenAI completion')

        content = content.strip()
        parsed = json.loads(content)

        return normalize_ai_params(parsed, video_metadata, style_key)
    except json.JSONDecodeError as jde:
        return {"error": f"AI response is not valid JSON: {jde}. Raw output: {content!r}"}
    except Exception as e:
        return {"error": str(e)}


def run_mosh(input_path: str, output_path: str, params: dict, job_id: str, style_name: str = '') -> dict:
    effect = params.get('effect', 'iframe_removal')
    start_frame = int(params.get('start_frame', 0))
    end_frame = int(params.get('end_frame', -1))
    fps = max(1, min(120, int(params.get('fps', 30))))
    _, style = resolve_style_preset(style_name)
    delta = int(params.get('delta', style.get('default_delta', 8))) if effect == 'delta_repeat' else 0
    boosted_delta = delta
    if delta > 0:
        boosted_delta = int(round(delta * float(style.get('delta_boost', 1.0))))
        boosted_delta = max(1, min(boosted_delta, 30))

    input_avi = f'tmp_{job_id}_in.avi'
    output_avi = f'tmp_{job_id}_out.avi'

    try:
        # Step 1: convert input to AVI (required for frame byte manipulation)
        ret = subprocess.call([
            ffmpeg_cmd(), '-loglevel', 'error', '-y',
            '-i', input_path,
            '-crf', '0', '-pix_fmt', 'yuv420p', '-bf', '0',
            '-b', '10000k', '-r', str(fps),
            input_avi
        ])
        if ret != 0 or not os.path.exists(input_avi):
            return {'success': False, 'error': 'ffmpeg AVI conversion failed. Is ffmpeg installed?'}

        # Step 2: read raw bytes and split into frames
        with open(input_avi, 'rb') as f:
            data = f.read()

        frame_marker = bytes.fromhex('30306463')  # '00dc' — end-of-frame marker
        iframe_sig = bytes.fromhex('0001B0')       # MPEG i-frame signature
        pframe_sig = bytes.fromhex('0001B6')       # MPEG p-frame (delta) signature

        frames = data.split(frame_marker)
        header = frames[0]
        frames = frames[1:]

        n_video_frames = sum(
            1 for f in frames
            if len(f) > 8 and (f[5:8] == iframe_sig or f[5:8] == pframe_sig)
        )
        if end_frame < 0:
            end_frame = n_video_frames

        # Step 3: apply datamoshing effect
        with open(output_avi, 'wb') as out:
            out.write(header)

            if boosted_delta > 0:
                # Delta repeat: capture N p-frames then loop them
                if boosted_delta > max(1, end_frame - start_frame):
                    return {
                        'success': False,
                        'error': f'Delta ({boosted_delta}) is larger than the frame range ({end_frame - start_frame}). '
                                 f'Try a smaller delta or a wider frame range.'
                    }

                repeat_frames = []
                repeat_index = 0

                for index, frame in enumerate(frames):
                    is_video = len(frame) > 8 and (frame[5:8] == iframe_sig or frame[5:8] == pframe_sig)
                    in_range = start_frame <= index < end_frame

                    if not is_video or not in_range:
                        out.write(frame_marker + frame)
                        continue
                    if (
                        style.get('drop_iframes_in_delta')
                        and len(frame) > 8
                        and frame[5:8] == iframe_sig
                    ):
                        continue

                    if len(repeat_frames) < boosted_delta and frame[5:8] != iframe_sig:
                        repeat_frames.append(frame)
                        out.write(frame_marker + frame)
                    elif len(repeat_frames) == boosted_delta:
                        out.write(frame_marker + repeat_frames[repeat_index])
                        repeat_index = (repeat_index + 1) % boosted_delta
                    else:
                        out.write(frame_marker + frame)
            else:
                # I-frame removal: drop keyframes so previous frame bleeds through
                for index, frame in enumerate(frames):
                    skip = (
                        len(frame) > 8
                        and frame[5:8] == iframe_sig
                        and start_frame <= index <= end_frame
                    )
                    if not skip:
                        out.write(frame_marker + frame)

        # Step 4: convert output AVI to mp4
        ret = subprocess.call([
            ffmpeg_cmd(), '-loglevel', 'error', '-y',
            '-i', output_avi,
            '-crf', str(style['output_crf']), '-pix_fmt', 'yuv420p',
            '-vcodec', 'libx264', '-acodec', 'aac',
            '-b', '10000k', '-r', str(fps),
            output_path
        ])

        if os.path.exists(output_path):
            return {'success': True}
        return {'success': False, 'error': 'Output file was not created. ffmpeg may have failed.'}

    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        for tmp in [input_avi, output_avi]:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass


def run_mosh_job(job_id: str, input_path: str, output_path: str, params: dict):
    processing_jobs[job_id] = {
        'status': 'processing',
        'progress': 0,
        'error': None,
        'params': params
    }

    def worker():
        try:
            result = run_mosh(input_path, output_path, params, job_id)
            if result.get('success'):
                processing_jobs[job_id]['status'] = 'completed'
                processing_jobs[job_id]['progress'] = 100
            else:
                processing_jobs[job_id]['status'] = 'failed'
                processing_jobs[job_id]['error'] = result.get('error', 'Unknown error')
                processing_jobs[job_id]['progress'] = 100
        except Exception as e:
            processing_jobs[job_id]['status'] = 'failed'
            processing_jobs[job_id]['error'] = str(e)
            processing_jobs[job_id]['progress'] = 100

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


def safe_remove(path: str):
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def schedule_delete(path: str, delay_seconds: int = 900):
    """Delete a file after a delay as a retention fallback."""
    def worker():
        time.sleep(max(0, int(delay_seconds)))
        safe_remove(path)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('videos', filename)


@app.route('/api/generated-image/<path:filename>')
def serve_generated_image(filename):
    safe_name = ''.join(c for c in filename if c.isalnum() or c in {'-', '_', '.'})
    if not safe_name:
        return jsonify({'error': 'Invalid filename'}), 400
    image_path = os.path.join(OUTPUT_FOLDER, safe_name)
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    return send_from_directory(OUTPUT_FOLDER, safe_name)


@app.route('/api/process', methods=['POST'])
def process():
    upload = request.files.get('video') or request.files.get('media') or request.files.get('image')
    if not upload or upload.filename == '':
        return jsonify({'error': 'No media file provided'}), 400

    media_type = detect_media_type(upload)
    if media_type == 'unknown':
        return jsonify({'error': 'Unsupported file type. Please upload a video or image.'}), 400

    job_id = str(uuid.uuid4())[:8]

    default_ext = '.mp4' if media_type == 'video' else '.jpg'
    ext = os.path.splitext(upload.filename)[1] or default_ext
    input_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_input{ext}')
    output_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_moshed.mp4')
    upload.save(input_path)

    use_manual = request.form.get('use_manual') == 'true'
    style_preset = request.form.get('style_preset', '').strip().lower()
    api_key = get_server_api_key()

    prompt = request.form.get('prompt', '').strip()
    params = {}

    if media_type == 'video':
        if use_manual:
            effect = request.form.get('effect', 'iframe_removal')
            params = {
                'effect': effect,
                'start_frame': int(request.form.get('start_frame', 0)),
                'end_frame': int(request.form.get('end_frame', -1)),
                'fps': int(request.form.get('fps', 30)),
                'delta': int(request.form.get('delta', 8)),
                'explanation': ''
            }
        else:
            if not prompt:
                safe_remove(input_path)
                return jsonify({'error': 'Please enter a description of the effect.'}), 400
            if not api_key:
                safe_remove(input_path)
                return jsonify({'error': 'Server OpenAI API key is missing. Set OPENAI_API_KEY and restart.'}), 400

            video_metadata = get_video_metadata(input_path)
            params = parse_prompt_with_ai(prompt, api_key, video_metadata, style_preset)
            if 'error' in params:
                safe_remove(input_path)
                return jsonify({'error': f'AI error: {params["error"]}'}), 400

        # Run mosh processing synchronously so client UI has immediate access to output
        result = run_mosh(input_path, output_path, params, job_id, style_preset)
        if not result.get('success'):
            safe_remove(input_path)
            return jsonify({'success': False, 'error': result.get('error', 'Processing failed.')}), 500

    response = {
        'success': True,
        'job_id': job_id,
        'media_type': media_type,
        'params': params,
        'explanation': params.get('explanation', '') if media_type == 'video' else ''
    }
    if media_type == 'video':
        response['download_url'] = f'/api/download/{job_id}'

    fallback_influence = (
        "I watched the clip and felt motion, fragmentation, and unresolved momentum."
        if media_type == 'video'
        else "I studied the image and felt stillness charged with symbolic tension."
    )
    fallback_narrative = (
        "I dreamed I was walking through your video as if it were a corridor of repeating light. "
        "Every cut felt like a memory trying to become a feeling, and the glitches moved like emotion without language. "
        "What part of this dream felt most familiar to you?"
        if media_type == 'video'
        else
        "I dreamed I stepped into your image and found it breathing in quiet cycles. "
        "Its details felt like clues from a memory that wanted to be understood. "
        "What part of this still moment feels most alive to you?"
    )
    response['dream'] = {
        'video_influence': fallback_influence,
        'initial_narrative': fallback_narrative,
        'suggested_image_prompt': fallback_narrative
    }

    if api_key:
        media_influence = (
            describe_video_content(input_path, api_key, job_id)
            if media_type == 'video'
            else describe_image_content(input_path, api_key)
        )
        video_influences[job_id] = media_influence

        conversation_histories[job_id] = [
            {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
            {"role": "system", "content": f"Video influence context: {media_influence}"}
        ]

        initial_seed = build_initial_dream_prompt(prompt if not use_manual else '')
        initial_narrative, conversation_histories[job_id] = get_narrative_response(
            initial_seed, conversation_histories[job_id], api_key
        )

        response['dream'] = {
            'video_influence': media_influence,
            'initial_narrative': initial_narrative,
            'suggested_image_prompt': narrative_to_image_prompt(job_id, fallback_prompt=media_influence)
        }

    # Remove uploaded source media after analysis is complete.
    safe_remove(input_path)
    # Retention fallback: if user never clicks download, remove processed video after 15 minutes.
    if media_type == 'video':
        schedule_delete(output_path, delay_seconds=900)
    return jsonify(response)


@app.route('/api/status/<job_id>')
def status(job_id):
    job_id = ''.join(c for c in job_id if c.isalnum() or c == '-')
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    status_info = processing_jobs[job_id].copy()
    output_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_moshed.mp4')
    status_info['downloadable'] = os.path.exists(output_path)
    status_info['download_url'] = f'/api/download/{job_id}' if status_info['downloadable'] else None

    return jsonify(status_info)


@app.route('/api/download/<job_id>')
def download(job_id):
    # Sanitize job_id to prevent path traversal
    job_id = ''.join(c for c in job_id if c.isalnum() or c == '-')
    output_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_moshed.mp4')
    if os.path.exists(output_path):
        response = send_file(output_path, as_attachment=True, download_name='moshed.mp4')
        # Delete processed output after serving download to minimize retained media.
        response.call_on_close(lambda: safe_remove(output_path))
        return response
    return jsonify({'error': 'File not found'}), 404


@app.route('/api/narrative', methods=['POST'])
def narrative():
    """Generate narrative response based on user input and maintain conversation history."""
    data = request.get_json()
    job_id = data.get('job_id', str(uuid.uuid4())[:8])
    user_input = data.get('message', '').strip()
    api_key = get_server_api_key()
    use_history = data.get('use_history', True)

    if not user_input:
        user_input = "Continue the dream."
    if not api_key:
        return jsonify({'error': 'Server OpenAI API key is missing. Set OPENAI_API_KEY and restart.'}), 400

    # Initialize or retrieve conversation history
    if job_id not in conversation_histories or not use_history:
        conversation_histories[job_id] = [
            {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT}
        ]
        if video_influences.get(job_id):
            conversation_histories[job_id].append(
                {"role": "system", "content": f"Video influence context: {video_influences[job_id]}"}
            )

    narrative_response, conversation_histories[job_id] = get_narrative_response(
        user_input, conversation_histories[job_id], api_key
    )

    return jsonify({
        'success': True,
        'job_id': job_id,
        'narrative': narrative_response,
        'suggested_image_prompt': narrative_to_image_prompt(job_id)
    })


@app.route('/api/narrative-media', methods=['POST'])
def narrative_media():
    """Continue narrative using optional text plus an uploaded image/video reference."""
    job_id = (request.form.get('job_id', str(uuid.uuid4())[:8]) or '').strip()
    user_input = (request.form.get('message', '') or '').strip()
    use_history = (request.form.get('use_history', 'true') or 'true').lower() == 'true'
    upload = request.files.get('media') or request.files.get('video') or request.files.get('image')
    api_key = get_server_api_key()

    if not api_key:
        return jsonify({'error': 'Server OpenAI API key is missing. Set OPENAI_API_KEY and restart.'}), 400

    media_context = ''
    if upload and upload.filename:
        media_type = detect_media_type(upload)
        if media_type == 'unknown':
            return jsonify({'error': 'Unsupported media type. Upload an image or video.'}), 400

        ext = os.path.splitext(upload.filename)[1] or ('.mp4' if media_type == 'video' else '.jpg')
        media_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_narrative_ref{ext}')
        upload.save(media_path)
        try:
            media_context = describe_media_content(media_path, media_type, api_key, job_id)
        finally:
            safe_remove(media_path)

        if media_context:
            media_label = 'video' if media_type == 'video' else 'image'
            if user_input:
                user_input = (
                    f"{user_input}\n\n"
                    f"Reference {media_label} context to weave in:\n{media_context}"
                )
            else:
                user_input = f"Continue the dream using this {media_label} reference:\n{media_context}"

    if not user_input:
        user_input = "Continue the dream."

    # Initialize or retrieve conversation history
    if job_id not in conversation_histories or not use_history:
        conversation_histories[job_id] = [
            {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT}
        ]
        if video_influences.get(job_id):
            conversation_histories[job_id].append(
                {"role": "system", "content": f"Video influence context: {video_influences[job_id]}"}
            )

    narrative_response, conversation_histories[job_id] = get_narrative_response(
        user_input, conversation_histories[job_id], api_key
    )

    return jsonify({
        'success': True,
        'job_id': job_id,
        'narrative': narrative_response,
        'suggested_image_prompt': narrative_to_image_prompt(job_id),
        'media_context': media_context
    })


@app.route('/api/generate-image', methods=['POST'])
def generate_image_endpoint():
    """Generate an image and an estimated movement matrix from the same prompt."""
    data = request.get_json()
    job_id = data.get('job_id', '').strip()
    prompt = data.get('prompt', '').strip()
    matrix_size = data.get('matrix_size', 8)
    image_count = data.get('image_count', 1)
    api_key = get_server_api_key()

    try:
        image_count = int(image_count)
    except (TypeError, ValueError):
        image_count = 1
    image_count = max(1, min(4, image_count))

    if not prompt:
        prompt = narrative_to_image_prompt(job_id, fallback_prompt=video_influences.get(job_id, ''))
    if not prompt:
        return jsonify({'error': 'Please provide a prompt for image generation.'}), 400
    if not api_key:
        return jsonify({'error': 'Server OpenAI API key is missing. Set OPENAI_API_KEY and restart.'}), 400

    # Enhance the prompt for photorealistic generation anchored to video-frame context.
    frame_context = video_influences.get(job_id, '').strip()
    enhanced_prompt = (
        "Create a photorealistic cinematic still that looks like a real photograph from a live-action scene. "
        "Prioritize realism: natural skin texture, plausible lighting, true-to-life materials, and camera-consistent depth of field. "
        "Do not render as painting, illustration, CGI, anime, or abstract art. "
        "If surreal elements are present, depict them with physically believable detail as if captured by a real camera. "
        f"Dream narrative direction: {prompt} "
        f"Video frame cues to ground composition, setting, and mood: {frame_context or 'Use realistic visual continuity with the uploaded video.'}"
    )
    movement_matrix = infer_movement_matrix(enhanced_prompt, api_key, grid_size=matrix_size)
    base_job_id = job_id or str(uuid.uuid4())[:8]
    images = []

    for idx in range(image_count):
        image_url = generate_image(enhanced_prompt, api_key)
        if "Error" in image_url:
            return jsonify({'error': image_url}), 500

        datamosh_ok, final_image_url = datamosh_generated_image(image_url, f'{base_job_id}_{idx + 1}')
        images.append({
            'image_url': final_image_url,
            'original_image_url': image_url,
            'datamosh_applied': datamosh_ok
        })

    first_image = images[0]

    return jsonify({
        'success': True,
        'image_url': first_image['image_url'],
        'original_image_url': first_image['original_image_url'],
        'datamosh_applied': first_image['datamosh_applied'],
        'image_count': image_count,
        'images': images,
        'prompt': prompt,
        'movement_matrix': movement_matrix
    })


@app.route('/api/clear-history/<job_id>', methods=['POST'])
def clear_history(job_id):
    """Clear conversation history for a job."""
    if job_id in conversation_histories:
        del conversation_histories[job_id]
    if job_id in video_influences:
        del video_influences[job_id]
    return jsonify({'success': True})


@app.route('/api/ping')
def ping():
    ffmpeg_ok = False
    try:
        ffmpeg_ok = subprocess.call(
            ['ffmpeg', '-version'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        ) == 0
    except FileNotFoundError:
        ffmpeg_ok = False
    except Exception:
        ffmpeg_ok = False

    return jsonify({
        'ok': True,
        'ffmpeg': ffmpeg_ok,
        'ffmpeg_path': 'ffmpeg',
        'hint': user_ffmpeg_install_hint(),
        'openai_configured': bool(get_server_api_key())
    })


@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({'error': 'Uploaded video is too large. Limit is 500 MB.'}), 413


@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Resource not found'}), 404


if __name__ == '__main__':
    # Quick startup checks
    print('\n=== Datamosh Studio ===')
    try:
        ret = subprocess.call(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if ret == 0:
            print('ffmpeg found')
        else:
            print('ffmpeg not found. Hint:', user_ffmpeg_install_hint())
    except FileNotFoundError:
        print('ffmpeg not found. Hint:', user_ffmpeg_install_hint())
    except Exception as e:
        print('ffmpeg check error:', e, '| Hint:', user_ffmpeg_install_hint())

    host = '0.0.0.0'
    port = 5000
    if is_port_in_use('127.0.0.1', port):
        print(f'Cannot start: port {port} is already in use.')
        print('Close the other server/process, then try again.')
        print('Windows helpers:')
        print('  netstat -ano | findstr :5000')
        print('  Stop-Process -Id <PID> -Force')
        sys.exit(1)

    print(f'Open http://127.0.0.1:{port}')
    app.run(debug=False, host=host, port=port)
