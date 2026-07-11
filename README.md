# 🌊 WaveCheck --- Audio Transcriber

WaveCheck is a production-ready Flask web application and REST API that
transcribes audio files into timestamped text using OpenAI's Whisper AI.
The application is designed to handle concurrent uploads, long audio
recordings, asynchronous processing, and failure recovery while
providing a modern web interface.

------------------------------------------------------------------------

# 🚀 Features

-   🎙️ **Speech-to-Text Transcription**
    -   Uses OpenAI Whisper (Base Model) to convert speech into text.
    -   Generates accurate timestamps for every spoken segment.
-   ✂️ **Automatic Audio Chunking**
    -   Files longer than **10 minutes** are automatically split into
        **5-minute chunks**.
    -   A **3-second overlap** prevents words from being cut between
        chunks.
    -   Timestamps are adjusted so they match the original audio.
-   ⚡ **Concurrent Processing**
    -   Uses `ThreadPoolExecutor` to process multiple transcription jobs
        simultaneously.
    -   Prevents the Flask server from blocking during long-running
        tasks.
-   🔄 **Retry Failed Jobs**
    -   Original uploaded audio files are preserved.
    -   Failed transcriptions can be retried without uploading the file
        again.
    -   Maximum retry limit is configurable.
-   💾 **Persistent Job Storage**
    -   Job status and transcription results are stored as JSON files.
    -   Completed jobs remain accessible even after server restarts.
-   🌐 **REST API**
    -   Asynchronous API using the Job Pattern.
    -   Upload files, check job status, retrieve results, and retry
        failed jobs.
-   🎨 **Modern User Interface**
    -   Drag & Drop audio upload
    -   Progress indicator
    -   Automatic polling
    -   Transcript viewer
    -   Timestamped segments
    -   Download transcript as TXT or JSON

------------------------------------------------------------------------

# 🛠️ Technologies Used

  Component          Technology
  ------------------ --------------------------------
  Backend            Flask
  Language           Python 3.8+
  AI Model           OpenAI Whisper
  Audio Processing   Pydub + FFmpeg
  Concurrency        ThreadPoolExecutor, threading
  Storage            JSON files + Local File System

------------------------------------------------------------------------

# 📦 Requirements

-   Python 3.8 or above
-   FFmpeg installed
-   pip

------------------------------------------------------------------------

# ⚙️ Installation

``` bash
git clone https://github.com/your-username/wavecheck.git
cd wavecheck

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install flask openai-whisper pydub

python app.py
```

Open your browser:

http://127.0.0.1:5000

------------------------------------------------------------------------

# 📂 Supported Audio Formats

-   WAV
-   MP3
-   M4A
-   OGG
-   FLAC

Maximum upload size: **500 MB**

------------------------------------------------------------------------

# 🔌 REST API

## POST /upload

Uploads an audio file and returns a queued job ID.

## GET /status/`<job_id>`{=html}

Returns the current status of a transcription job.

## GET /result/`<job_id>`{=html}

Returns the completed transcript and timestamped segments.

## POST /retry/`<job_id>`{=html}

Retries a failed transcription using the original uploaded audio.

------------------------------------------------------------------------

# 🏗️ Design Decisions

-   Background processing using **ThreadPoolExecutor**
-   Long audio chunking with overlapping segments
-   Persistent JSON job storage
-   Retry mechanism using original uploaded files
-   Asynchronous REST API

------------------------------------------------------------------------

# ⚙️ Configuration

  Variable             Default
  -------------------- ---------
  MAX_CONTENT_LENGTH   500 MB
  MAX_WORKERS          2
  CHUNK_THRESHOLD_MS   10 min
  CHUNK_SIZE_MS        5 min
  OVERLAP_MS           3000 ms
  RETRY_LIMIT          3

------------------------------------------------------------------------

# 📁 Project Structure

``` text
wavecheck/
├── app.py
├── uploads/
├── jobs/
└── README.md
```

------------------------------------------------------------------------

# 👨‍💻 Author

Developed as part of the WaveCheck Audio Transcriber project using
Flask, OpenAI Whisper, and Python.
