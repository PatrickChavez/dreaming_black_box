# Dreaming Black Box

An AI that turns your day into a dream.

Feed it images, videos, and notes from your day. Tell it a bedtime story. It will generate a written dream recall, a series of AI images, a voice narration, and a datamoshed video shaped by what you gave it and the emotional tone it detected.

---

## What it does

1. **Ingest memories** — Drop in images, videos, or text files from your day. The AI describes each one using GPT-4o-mini vision and stores the descriptions as "memories" for the session.
2. **Bedtime story** — Write anything: a short story, a description of your mood, a sentence. The AI reads the tone (calm / surreal / horror / nostalgic) and uses it to steer the dream.
3. **Dream recall** — GPT-4o generates a structured dream narrative from your memories and story, then writes a first-person waking monologue.
4. **Dream visuals + video** — DALL-E 3 generates up to 6 images. Each image is animated with a slow zoom, datamoshed to create glitch artifacts, and assembled into a video with a TTS voice narration.

---

## Requirements

- Python 3.11+
- ffmpeg (must be on your PATH)
- An OpenAI API key

**Install ffmpeg:**
- Mac: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`
- Windows: download from ffmpeg.org, add the `bin/` folder to PATH

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd Datamoshing-main

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements_webapp.txt

# 4. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."  # Windows: set OPENAI_API_KEY=sk-...

# 5. Run the server
python app.py
```

Then open **http://127.0.0.1:5001** in your browser.

---

## Cost estimate

All AI calls go through your OpenAI API key. Rough costs per session:

| Step | Model | Cost |
|---|---|---|
| Memory captioning | GPT-4o-mini Vision | ~$0.01–0.05 per file |
| Tone classification | GPT-4o-mini | < $0.001 |
| Dream narrative | GPT-4o | ~$0.03–0.05 |
| Recall text | GPT-4o | ~$0.02 |
| Images | DALL-E 3 HD | $0.04 per image |
| TTS narration | TTS-1-HD | ~$0.01–0.03 |

A full session with 3 images costs roughly **$0.20–0.30**.

---

## Project structure

```
app.py                  Main Flask server — all backend logic lives here
templates/
  dream.html            Single-page frontend (vanilla JS + Tailwind)
videos/
  BlackBoxBG.mp4        Looping background video for the UI
uploads/                Temporary storage for uploaded files (cleared on exit)
outputs/                Generated images, audio, and video (cleared on exit)
memory_store/           Session memories persisted as JSON (kept on exit)
requirements_webapp.txt Python dependencies
```

---

## How the datamosh works

Video files are made up of two kinds of frames:
- **I-frames** (keyframes) — a complete image, self-contained
- **P-frames** (delta frames) — only the *difference* from the previous frame

Datamoshing works by manipulating these frames at the binary level, inside the raw AVI container:

- **I-frame removal** — strip keyframes so the decoder has nothing to reference. Motion vectors from the previous scene bleed forward, smearing one image into the next.
- **Delta repeat** — capture a handful of P-frames and loop them, creating a stuttering frozen-motion effect.

The `run_mosh()` function in `app.py` does this by splitting the AVI byte stream on the frame marker `30306463`, identifying I-frames by the signature `0001B0` and P-frames by `0001B6`, rewriting the stream, then re-encoding to MP4.

REF: https://github.com/tiberiuiancu/datamoshing

---

## Notes

- **On shutdown** (Ctrl+C), the server automatically wipes `outputs/` and `uploads/`. Session memories in `memory_store/` are kept.
- Generated files are also auto-deleted after **2 hours** while the server is running.
- The server runs on port **5001**. If that port is taken, the app will tell you and exit.
- The API key is read from the environment. Never hardcode it in the source.
