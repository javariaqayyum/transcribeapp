# AudioScript — Speech-to-Text Transcription Pipeline

A Flask web application that transcribes audio files using OpenAI Whisper.

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg (for audio format conversion)
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# Mac
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

## Features
- Drag & drop or click to upload audio
- Supports WAV, MP3, M4A, OGG, FLAC, MP4, WEBM
- Live audio preview before transcribing
- Full transcript + timestamped segments
- Download as .txt or .json
- Auto language detection

## Whisper Models
Edit `app.py` line `model = whisper.load_model("base")` to change quality:
- `tiny`  — fastest, least accurate
- `base`  — good balance (default)
- `small` — better accuracy
- `medium`— high accuracy
- `large` — best accuracy, slowest
