# DREAMING BLACK BOX — Integration Guide

## Overview

Your `app.py` and `cart498_final_6.py` have been combined into a unified **DREAMING BLACK BOX** — an ELIZA-style interactive application that:

1. **Processes Videos** with datamoshing glitch effects (iframe_removal or delta_repeat)
2. **Generates Narratives** based on the video, using AI to weave dream-like stories
3. **Maintains Conversation History** with optional toggle for persistent multi-turn dialogue
4. **Generates Dream Images** from the narrative using DALL-E
5. **Adds Typewriter Effects** for immersive storytelling

---

## What Changed

### Backend (`app.py`)

✅ **New Functions:**

- `get_narrative_response()` — Generates narrative text with conversation history
- `generate_image()` — Creates visual representations using DALL-E
- `describe_video_content()` — Placeholder for future AI video analysis

✅ **New API Endpoints:**

- `/api/narrative` — POST: Generate/continue narrative responses
  - Parameters: `job_id`, `message`, `api_key`, `use_history` (boolean)
- `/api/generate-image` — POST: Create images from narrative
  - Parameters: `job_id`, `prompt`, `api_key`
- `/api/clear-history/<job_id>` — POST: Reset conversation history
- Conversation histories stored in-memory per `job_id`

✅ **Configuration:**

- `NARRATIVE_SYSTEM_PROMPT` — Controls storytelling tone (whimsical, dreamlike, ELIZA-style)
- All existing datamoshing features remain unchanged

### Frontend (`index.html`)

✅ **New UI Sections:**

- **Narrative Display** — Typewriter effect text display with dreamlike styling
- **Conversation Input** — Continue narrative with optional history toggle
- **Image Generation** — Textarea for custom image prompts + display area
- **Reset Button** — Clear conversation history per session

✅ **Features:**

- Typewriter animation (30ms per character, adjustable)
- Real-time history toggle (checkbox: "Keep history")
- Pre-filled image prompt from last narrative response
- Download button for generated images
- Gradient background and immersive styling

---

## Workflow

```
1. Upload Video
   ↓
2. Choose Mode (AI or Manual)
   ↓
3. Describe the Video/Glitch Effect
   ↓
4. AI Processes Video (datamoshing effect applied)
   ↓
5. Video Download Available
   ↓
6. Enter Dream Description or Prompt (AI-generated narrative shown with typewriter effect)
   ↓
7. Continue Narrative (multi-turn conversation with optional history)
   ↓
8. Generate Dream Image (from narrative text)
   ↓
9. Download Generated Image
```

---

## Usage Instructions

### Running the App

```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

### Features in Detail

#### 1. Video Upload & Processing

- **AI Mode**: Describe the effect ("glitchy transition in the middle", "melting effect")
- **Manual Mode**: Specify effect type, frame ranges, FPS, delta parameters
- Video processing uses existing datamoshing engine (no changes)

#### 2. Narrative Generation

- **First Input**: Describe your dream or video subject
- **Continue Narrative**: Add more details, ask AI to expand, or redirect the story
- **History Toggle**: Check "Keep history" to maintain conversation; uncheck for fresh start
- **Typewriter Effect**: Text appears character-by-character for immersion

#### 3. Image Generation

- Textarea pre-filled with last narrative response (editable)
- Click "Generate Image" to create DALL-E visualization
- Images are surreal + dreamlike (enhanced prompt)
- Download link provided

---

## Configuration & Customization

### Narrative Tone

Edit `NARRATIVE_SYSTEM_PROMPT` in `app.py`:

```python
NARRATIVE_SYSTEM_PROMPT = """You are a whimsical, imaginative storyteller..."""
```

### Typewriter Speed

In `index.html`, adjust `delay` parameter (milliseconds per character):

```javascript
typewriterDisplay(text, 30); // 30ms = faster; 100ms = slower
```

### Image Enhancement

In `/api/generate-image` endpoint, customize the prompt enhancement:

```python
enhanced_prompt = f"A surreal, dreamlike, artistic visualization of: {prompt}"
```

---

## API Reference

### POST `/api/process`

_Existing endpoint — unchanged_
Video datamoshing processing

### POST `/api/narrative`

Generate narrative response with conversation history

```json
{
  "job_id": "abc123",
  "message": "I dreamt I was flying",
  "api_key": "sk-...",
  "use_history": true
}
```

**Response:**

```json
{
  "success": true,
  "job_id": "abc123",
  "narrative": "You soared above clouds..."
}
```

### POST `/api/generate-image`

Generate image from narrative

```json
{
  "job_id": "abc123",
  "prompt": "Surreal flying through purple clouds",
  "api_key": "sk-..."
}
```

**Response:**

```json
{
  "success": true,
  "image_url": "https://oaidalleapiprodpuc.blob.core.windows.net/...",
  "prompt": "Surreal flying through purple clouds"
}
```

### POST `/api/clear-history/<job_id>`

Reset conversation history

```json
{
  "success": true
}
```

---

## Requirements

No new dependencies added! Uses existing:

- `flask>=3.0.0`
- `openai>=1.0.0`

Existing system requirements still apply:

- `ffmpeg` (for video processing)
- Valid OpenAI API key (with DALL-E access for image generation)

---

## Notes

### Conversation History

- Stored in-memory per session (not persistent across server restarts)
- Toggle per job (different videos can have independent histories)
- Clearing history removes all previous context for that job

### Image Generation

- Requires DALL-E API access (standard OpenAI plan)
- Note: DALL-E 3 usage incurs additional costs
- Images are 1024x1024, standard quality

### Typewriter Effect

- Client-side only (no server overhead)
- Disabled during image generation loading states

---

## Troubleshooting

**Issue**: "API key is required"

- Solution: Paste your OpenAI API key in the settings section

**Issue**: Image generation fails with "Error generating image"

- Check: DALL-E access enabled in your OpenAI account
- Check: API key has sufficient credits

**Issue**: Narrative doesn't continue conversation

- Check: "Keep history" checkbox is enabled
- Try: Clicking "Reset" and starting a new conversation

**Issue**: ffmpeg not found

- Solution: `brew install ffmpeg` (macOS) or install via your package manager

---

## Files Modified

1. **app.py** — Added narrative, image generation, and conversation history endpoints
2. **templates/index.html** — New UI sections for narrative display, conversation, and image generation
3. **INTEGRATION_GUIDE.md** — This file

---

Enjoy your DREAMING BLACK BOX! 🎬✨
