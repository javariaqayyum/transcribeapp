from flask import Flask, request, jsonify, render_template_string
import os, uuid, shutil, whisper, json, threading 
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor             
from datetime import datetime 

os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

SUPPORTED_FORMATS    = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}
UPLOAD_FOLDER        = os.path.join(os.getcwd(), "uploads")
CHUNK_THRESHOLD_MS   = 10 * 60 * 1000   # files > 10 min get chunked
CHUNK_SIZE_MS        = 5  * 60 * 1000   # 5-minute chunks
OVERLAP_MS           = 3000             # 3-sec overlap to avoid cutting sentences


JOBS_FOLDER   = os.path.join(os.getcwd(), "jobs")
MAX_WORKERS   = 2          
RETRY_LIMIT   = 3          

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(JOBS_FOLDER, exist_ok=True)

print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model ready.")


executor  = ThreadPoolExecutor(max_workers=MAX_WORKERS)
jobs      = {}          # {job_id: {status, filename, result, error, attempts, ...}}
jobs_lock = threading.Lock()


# LONG FILE HANDLER — split → transcribe → merge

def split_audio(audio, chunk_ms, overlap_ms):
    """Split audio into overlapping chunks. Returns list of (chunk, offset_sec)."""
    chunks = []
    start  = 0
    while start < len(audio):
        end   = min(start + chunk_ms + overlap_ms, len(audio))
        chunk = audio[start:end]
        chunks.append((chunk, start / 1000))
        start += chunk_ms
    return chunks

def transcribe_in_chunks(audio, tmp_dir):
    """Transcribe long audio in 5-min chunks, correcting timestamps by offset."""
    chunks       = split_audio(audio, CHUNK_SIZE_MS, OVERLAP_MS)
    all_segments = []
    full_text    = []
    language     = 'unknown'

    print(f"Long file — splitting into {len(chunks)} chunks")

    for i, (chunk, offset_sec) in enumerate(chunks):
        chunk_path = os.path.join(tmp_dir, f"chunk_{i}.wav")
        chunk.set_frame_rate(16000).set_channels(1).export(chunk_path, format="wav")

        try:
            result   = model.transcribe(chunk_path, verbose=False)
            language = result.get('language', language)
            full_text.append(result['text'].strip())

            # Shift timestamps by chunk offset so they point to correct position
            for seg in result['segments']:
                all_segments.append({
                    'id':    len(all_segments),
                    'start': round(seg['start'] + offset_sec, 2),
                    'end':   round(seg['end']   + offset_sec, 2),
                    'text':  seg['text'].strip()
                })
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    return {
        'text':     ' '.join(full_text),
        'segments': all_segments,
        'language': language
    }
## Background Worker Function

def process_job(job_id, original_path, ext):
    """Transcribe in a background thread. Keeps original audio for retry."""
    with jobs_lock:
        jobs[job_id]['status'] = 'processing'

    work_dir = os.path.join(UPLOAD_FOLDER, job_id, "work")
    os.makedirs(work_dir, exist_ok=True)

    try:
        AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
        AudioSegment.ffprobe   = r"C:\ffmpeg\bin\ffprobe.exe"

        audio = AudioSegment.from_file(original_path)
        audio = audio.set_frame_rate(16000).set_channels(1)

        wav_path = os.path.join(work_dir, "converted.wav")
        audio.export(wav_path, format="wav")

        duration_ms = len(audio)
        chunked     = duration_ms > CHUNK_THRESHOLD_MS

        if chunked:
            print(f"[{job_id}] Long file ({duration_ms/60000:.1f} min) — chunking")
            result   = transcribe_in_chunks(audio, work_dir)
            segments = result['segments']
            language = result['language']
            text     = result['text']
        else:
            print(f"[{job_id}] Short file ({duration_ms/60000:.1f} min) — direct")
            result   = model.transcribe(wav_path, verbose=False)
            language = result.get('language', 'unknown')
            text     = result['text']
            segments = [
                {
                    'id':    i,
                    'start': round(seg['start'], 2),
                    'end':   round(seg['end'],   2),
                    'text':  seg['text'].strip()
                }
                for i, seg in enumerate(result['segments'])
            ]

        duration    = round(duration_ms / 1000, 1)
        result_data = {
            'transcript': text,
            'language':   language,
            'duration':   duration,
            'segments':   segments,
            'chunked':    chunked
        }

        # Update in-memory store
        with jobs_lock:
            jobs[job_id]['status']       = 'completed'
            jobs[job_id]['result']       = result_data
            jobs[job_id]['completed_at'] = datetime.now().isoformat()

        # Persist to disk — survives server restarts
        persist_path = os.path.join(JOBS_FOLDER, f"{job_id}.json")
        with open(persist_path, 'w', encoding='utf-8') as f:
            json.dump({
                'id':           job_id,
                'filename':     jobs[job_id]['filename'],
                'status':       'completed',
                'result':       result_data,
                'created_at':   jobs[job_id]['created_at'],
                'completed_at': jobs[job_id]['completed_at'],
                'attempts':     jobs[job_id]['attempts']
            }, f, ensure_ascii=False)

        print(f"[{job_id}]  Completed")

    except Exception as e:
        print(f"[{job_id}]  Failed: {e}")
        with jobs_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error']  = str(e)

        persist_path = os.path.join(JOBS_FOLDER, f"{job_id}.json")
        with open(persist_path, 'w', encoding='utf-8') as f:
            json.dump({
                'id':         job_id,
                'filename':   jobs[job_id]['filename'],
                'status':     'failed',
                'error':      str(e),
                'created_at': jobs[job_id]['created_at'],
                'attempts':   jobs[job_id]['attempts']
            }, f, ensure_ascii=False)

    finally:
        # Clean temp work files, but KEEP original audio for retry
        shutil.rmtree(work_dir, ignore_errors=True)

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WaveCheck — Audio Transcriber</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:         #F7F8FC;
  --white:      #FFFFFF;
  --border:     #E4E7EF;
  --text:       #1A1D2E;
  --muted:      #8A91A8;
  --accent:     #5B4CF5;
  --accent-lt:  #EEF0FF;
  --green:      #12B76A;
  --green-lt:   #ECFDF5;
  --green-bd:   #A6F4C5;
  --red:        #F04438;
  --red-lt:     #FEF3F2;
  --red-bd:     #FECDCA;
  --shadow-sm:  0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md:  0 4px 16px rgba(0,0,0,0.08), 0 2px 6px rgba(0,0,0,0.04);
  --radius:     16px;
  --sans:       'Sora', sans-serif;
  --mono:       'JetBrains Mono', monospace;
}

body {
  background: var(--bg); font-family: var(--sans); min-height: 100vh;
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 40px 20px;
}
body::before {
  content: ''; position: fixed; top: -200px; right: -200px;
  width: 600px; height: 600px;
  background: radial-gradient(circle, rgba(91,76,245,0.07) 0%, transparent 70%);
  pointer-events: none;
}
body::after {
  content: ''; position: fixed; bottom: -200px; left: -200px;
  width: 500px; height: 500px;
  background: radial-gradient(circle, rgba(18,183,106,0.06) 0%, transparent 70%);
  pointer-events: none;
}

.header { text-align: center; margin-bottom: 32px; }
.logo-wrap {
  display: inline-flex; align-items: center; justify-content: center;
  width: 56px; height: 56px; background: var(--accent);
  border-radius: 16px; margin-bottom: 16px;
  box-shadow: 0 8px 24px rgba(91,76,245,0.30);
}
.logo-wrap svg { width: 28px; height: 28px; fill: white; }
h1 { font-size: 28px; font-weight: 700; color: var(--text); letter-spacing: -0.5px; margin-bottom: 6px; }
.subtitle { font-size: 14px; color: var(--muted); font-weight: 400; }

.card {
  background: var(--white); border: 1px solid var(--border);
  border-radius: var(--radius); box-shadow: var(--shadow-md);
  padding: 36px; width: 100%; max-width: 500px;
}

#dropzone {
  border: 2px dashed var(--border); border-radius: 12px;
  padding: 44px 24px; text-align: center; cursor: pointer;
  position: relative; transition: border-color 0.2s, background 0.2s; background: var(--bg);
}
#dropzone:hover { border-color: var(--accent); background: var(--accent-lt); }
#dropzone.dragover { border-color: var(--accent); background: var(--accent-lt); transform: scale(1.01); }
#dropzone input[type="file"] {
  position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
}
.drop-icon {
  width: 48px; height: 48px; background: var(--white); border: 1px solid var(--border);
  border-radius: 12px; display: flex; align-items: center; justify-content: center;
  margin: 0 auto 14px; box-shadow: var(--shadow-sm); font-size: 22px; transition: transform 0.2s;
}
#dropzone:hover .drop-icon { transform: translateY(-3px); }
.drop-title { font-size: 15px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
.drop-sub   { font-size: 13px; color: var(--muted); margin-bottom: 14px; }
.format-pills { display: flex; gap: 6px; justify-content: center; flex-wrap: wrap; }
.pill {
  font-family: var(--mono); font-size: 10px; font-weight: 500;
  padding: 3px 8px; border-radius: 99px; background: var(--white);
  border: 1px solid var(--border); color: var(--muted); letter-spacing: 0.3px;
}

#file-row {
  display: none; margin-top: 16px; background: var(--bg);
  border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 14px; align-items: center; gap: 12px;
}
#file-row.show { display: flex; }
.file-icon-box {
  width: 38px; height: 38px; background: var(--accent-lt); border-radius: 8px;
  display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0;
}
.file-details { flex: 1; min-width: 0; }
.file-name {
  font-size: 13px; font-weight: 600; color: var(--text);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.file-size { font-size: 11px; color: var(--muted); margin-top: 2px; font-family: var(--mono); }
.clear-btn {
  background: none; border: none; cursor: pointer; color: var(--muted);
  font-size: 18px; padding: 4px; border-radius: 6px;
  transition: color 0.15s, background 0.15s; line-height: 1;
}
.clear-btn:hover { color: var(--red); background: var(--red-lt); }

#submit-btn {
  width: 100%; margin-top: 20px; padding: 14px; background: var(--accent);
  color: white; border: none; border-radius: 10px; font-size: 15px;
  font-weight: 600; font-family: var(--sans); cursor: pointer;
  transition: background 0.2s, transform 0.1s, box-shadow 0.2s;
  box-shadow: 0 4px 14px rgba(91,76,245,0.30);
  display: flex; align-items: center; justify-content: center; gap: 8px;
}
#submit-btn:hover:not(:disabled) {
  background: #4A3BE0; box-shadow: 0 6px 20px rgba(91,76,245,0.40); transform: translateY(-1px);
}
#submit-btn:active:not(:disabled) { transform: translateY(0); }
#submit-btn:disabled { background: #C4C6D4; box-shadow: none; cursor: not-allowed; }

#progress-wrap { display: none; margin-top: 16px; }
#progress-wrap.show { display: block; }
.progress-bg { height: 4px; background: var(--border); border-radius: 99px; overflow: hidden; margin-bottom: 8px; }
.progress-fill {
  height: 100%; width: 100%;
  background: linear-gradient(90deg, var(--accent), #9C91FF);
  border-radius: 99px; animation: shimmer 1.4s ease-in-out infinite;
}
@keyframes shimmer { 0%,100%{opacity:1} 50%{opacity:0.5} }
.progress-label { font-size: 12px; color: var(--muted); font-family: var(--mono); }

#toast {
  display: none; margin-top: 18px; padding: 14px 16px; border-radius: 10px;
  font-size: 14px; font-weight: 500; align-items: flex-start; gap: 10px;
  animation: slideUp 0.25s ease;
}
#toast.show    { display: flex; }
#toast.success { background: var(--green-lt); border: 1px solid var(--green-bd); color: #027A48; }
#toast.error   { background: var(--red-lt);   border: 1px solid var(--red-bd);   color: #B42318; }
.toast-icon    { font-size: 18px; flex-shrink: 0; line-height: 1; margin-top: 1px; }
.toast-body    { flex: 1; }
.toast-title   { font-weight: 700; margin-bottom: 2px; }
.toast-detail  { font-size: 12px; opacity: 0.8; font-family: var(--mono); }

#toast-retry {
  display: none; margin-top: 8px; padding: 5px 12px;
  background: var(--red); color: white; border: none;
  border-radius: 6px; font-size: 12px; font-weight: 600;
  font-family: var(--sans); cursor: pointer;
  transition: background 0.15s;
}
#toast-retry:hover { background: #D92C20; }
#toast-retry.show  { display: inline-block; }


#transcript-box {
  display: none; margin-top: 20px; padding: 16px; background: #F8F9FC;
  border: 1px solid var(--border); border-radius: 12px; animation: slideUp 0.3s ease;
}
#transcript-box.show { display: block; }
.transcript-header {
  display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;
}
.transcript-title { font-size: 13px; font-weight: 700; color: var(--text); text-transform: uppercase; letter-spacing: 0.5px; }
.transcript-actions { display: flex; gap: 6px; }
.t-btn {
  font-size: 11px; font-weight: 600; font-family: var(--sans);
  padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border);
  background: var(--white); color: var(--muted); cursor: pointer; transition: all 0.15s;
}
.t-btn:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-lt); }
.transcript-meta { display: flex; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }
.meta-tag {
  font-family: var(--mono); font-size: 10px; font-weight: 600;
  padding: 2px 8px; border-radius: 99px;
  background: var(--accent-lt); color: var(--accent); border: 1px solid #C7C2FB;
}
.transcript-text {
  font-size: 14px; line-height: 1.8; color: #33384F; white-space: pre-wrap;
  background: var(--white); padding: 14px; border-radius: 8px;
  border: 1px solid #E4E7EF; max-height: 200px; overflow-y: auto;
}

.segments-label {
  font-size: 11px; font-weight: 600; letter-spacing: 0.8px;
  text-transform: uppercase; color: var(--muted); margin: 16px 0 10px;
}
#segments-list {
  display: flex; flex-direction: column; gap: 6px;
  max-height: 320px; overflow-y: auto; padding-right: 4px;
}
.seg-row {
  display: grid; grid-template-columns: 110px 1fr; gap: 10px;
  background: var(--white); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 12px;
  transition: border-color 0.15s, box-shadow 0.15s; animation: slideUp 0.2s ease;
}
.seg-row:hover { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(91,76,245,0.07); }
.seg-time { display: flex; flex-direction: column; gap: 2px; padding-top: 1px; }
.seg-start { font-family: var(--mono); font-size: 11px; font-weight: 600; color: var(--accent); }
.seg-end   { font-family: var(--mono); font-size: 10px; color: var(--muted); }
.seg-text  { font-size: 13px; line-height: 1.55; color: var(--text); }

.divider { display: flex; align-items: center; gap: 12px; margin: 24px 0 0; }
.divider-line  { flex: 1; height: 1px; background: var(--border); }
.divider-label { font-size: 11px; color: var(--muted); font-weight: 500; letter-spacing: 0.5px; }
.formats-footer { margin-top: 10px; text-align: center; font-size: 12px; color: var(--muted); }
.formats-footer span { color: var(--text); font-weight: 600; }

@keyframes slideUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
@media (max-width: 540px) { .card { padding: 24px 18px; } h1 { font-size: 22px; } }
</style>
</head>
<body>

<div class="header">
  <div class="logo-wrap">
    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 3a1 1 0 0 1 1 1v16a1 1 0 0 1-2 0V4a1 1 0 0 1 1-1zM7 7a1 1 0 0 1 1 1v8a1 1 0 0 1-2 0V8a1 1 0 0 1 1-1zm10 0a1 1 0 0 1 1 1v8a1 1 0 0 1-2 0V8a1 1 0 0 1 1-1zM4 10a1 1 0 0 1 1 1v2a1 1 0 0 1-2 0v-2a1 1 0 0 1 1-1zm16 0a1 1 0 0 1 1 1v2a1 1 0 0 1-2 0v-2a1 1 0 0 1 1-1z"/>
    </svg>
  </div>
  <h1>WaveCheck</h1>
  <p class="subtitle">Audio Transcriber — drop a file to process it</p>
</div>

<div class="card">
  <div id="dropzone">
    <input type="file" id="fileInput" accept=".wav,.mp3,.m4a,.ogg,.flac">
    <div class="drop-icon">🎵</div>
    <div class="drop-title">Drop your audio file here</div>
    <div class="drop-sub">or click to browse files</div>
    <div class="format-pills">
      <span class="pill">WAV</span><span class="pill">MP3</span>
      <span class="pill">M4A</span><span class="pill">OGG</span><span class="pill">FLAC</span>
    </div>
  </div>

  <div id="file-row">
    <div class="file-icon-box">🎶</div>
    <div class="file-details">
      <div class="file-name" id="file-name"></div>
      <div class="file-size" id="file-size"></div>
    </div>
    <button class="clear-btn" id="clear-btn" title="Remove file">✕</button>
  </div>

  <button id="submit-btn" disabled>🎙 Transcribe File</button>

  <div id="progress-wrap">
    <div class="progress-bg"><div class="progress-fill"></div></div>
    <div class="progress-label" id="progress-label">Processing audio...</div>
  </div>

  <div id="toast">
    <div class="toast-icon" id="toast-icon"></div>
    <div class="toast-body">
      <div class="toast-title" id="toast-title"></div>
      <div class="toast-detail" id="toast-detail"></div>
      <button id="toast-retry">🔄 Retry Transcription</button>
    </div>
  </div>

  <div id="transcript-box">
    <div class="transcript-header">
      <div class="transcript-title">📄 Spoken Transcript</div>
      <div class="transcript-actions">
        <button class="t-btn" id="copy-btn">📋 Copy</button>
        <button class="t-btn" id="dl-btn">⬇ .txt</button>
        <button class="t-btn" id="dl-json-btn">⬇ .json</button>
      </div>
    </div>
    <div class="transcript-meta" id="transcript-meta"></div>
    <div class="transcript-text" id="transcript-text"></div>
    <div class="segments-label">⏱ Segments with Timestamps</div>
    <div id="segments-list"></div>
  </div>

  <div class="divider">
    <div class="divider-line"></div>
    <div class="divider-label">SUPPORTED FORMATS</div>
    <div class="divider-line"></div>
  </div>
  <div class="formats-footer">
    <span>WAV · MP3 · M4A · OGG · FLAC</span> &nbsp;·&nbsp; Max file size 500 MB
  </div>
</div>

<script>
  const fileInput      = document.getElementById('fileInput');
  const dropzone       = document.getElementById('dropzone');
  const fileRow        = document.getElementById('file-row');
  const fileNameEl     = document.getElementById('file-name');
  const fileSizeEl     = document.getElementById('file-size');
  const clearBtn       = document.getElementById('clear-btn');
  const submitBtn      = document.getElementById('submit-btn');
  const toast          = document.getElementById('toast');
  const toastIcon      = document.getElementById('toast-icon');
  const toastTitle     = document.getElementById('toast-title');
  const toastDetail    = document.getElementById('toast-detail');
  const toastRetry     = document.getElementById('toast-retry');
  const progressWrap   = document.getElementById('progress-wrap');
  const progressLabel  = document.getElementById('progress-label');
  const transcriptBox  = document.getElementById('transcript-box');
  const transcriptMeta = document.getElementById('transcript-meta');
  const transcriptText = document.getElementById('transcript-text');
  const copyBtn        = document.getElementById('copy-btn');
  const dlBtn          = document.getElementById('dl-btn');
  const dlJsonBtn      = document.getElementById('dl-json-btn');
  const segmentsList   = document.getElementById('segments-list');

  let selectedFile = null;
  let lastText     = '';
  let lastSegments = [];
  let currentJobId = null;    // track current job for retry
  let pollTimer    = null;    // polling interval reference

  function formatBytes(b) {
    if (b < 1024)    return b + ' B';
    if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
    return (b/1048576).toFixed(1) + ' MB';
  }
  function formatTime(sec) {
    const m  = Math.floor(sec / 60);
    const s  = Math.floor(sec % 60);
    const ms = Math.round((sec % 1) * 10);
    return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${ms}`;
  }

  function setFile(file) {
    selectedFile = file;
    fileNameEl.textContent = file.name;
    fileSizeEl.textContent = formatBytes(file.size) + ' · ' + (file.type || 'audio file');
    fileRow.classList.add('show');
    submitBtn.disabled = false;
    hideToast(); hideTranscript();
  }
  function clearFile() {
    selectedFile = null; fileInput.value = '';
    fileRow.classList.remove('show'); submitBtn.disabled = true;
    hideToast(); hideTranscript(); progressWrap.classList.remove('show');
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }
  function showToast(type, title, detail, showRetry) {             
    toast.className = 'show ' + type;
    toastIcon.textContent  = type === 'success' ? '✅' : '❌';
    toastTitle.textContent  = title;
    toastDetail.textContent = detail;
    toastRetry.className = showRetry ? 'show' : '';
    currentJobId = showRetry ? currentJobId : null;
  }
  function hideToast()      {  toast.className = ''; toastRetry.className = '';  }
  function hideTranscript() { transcriptBox.classList.remove('show'); segmentsList.innerHTML = ''; }

  fileInput.addEventListener('change', e => { if (e.target.files[0]) setFile(e.target.files[0]); });
  clearBtn.addEventListener('click',   e => { e.stopPropagation(); clearFile(); });
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault(); dropzone.classList.remove('dragover');
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  });

  const steps = [
    'Validating file...',
    'Converting audio format...',
    'Checking file length...',
    'Running Whisper AI...',
    'Building transcript...',
    'Almost done...'
  ];
  let stepIdx = 0, stepTimer = null;

// display result from pooling response

  function displayResult(data) {
    lastText     = data.transcript || data.result?.transcript || '';
    lastSegments = data.segments || data.result?.segments || [];
    const words  = lastText.trim().split(/\s+/).length;
    const lang   = (data.language || data.result?.language || 'unknown').toUpperCase();
    const dur    = data.duration || data.result?.duration || '?';
    const chunked= data.chunked || data.result?.chunked || false;

    transcriptMeta.innerHTML = `
      <span class="meta-tag">🌐 ${lang}</span>
      <span class="meta-tag">📝 ${words} words</span>
      <span class="meta-tag">⏱ ${dur}s</span>
      <span class="meta-tag">🔖 ${lastSegments.length} segments</span>
      <span class="meta-tag">${chunked ? '🔀 Chunked' : '⚡ Direct'}</span>
      <span class="meta-tag">🤖 Whisper AI</span>
    `;
    transcriptText.textContent = lastText.trim();

    segmentsList.innerHTML = '';
    lastSegments.forEach(seg => {
      const row = document.createElement('div');
      row.className = 'seg-row';
      row.innerHTML = `
        <div class="seg-time">
          <span class="seg-start">▶ ${formatTime(seg.start)}</span>
          <span class="seg-end">⏹ ${formatTime(seg.end)}</span>
        </div>
        <div class="seg-text">${seg.text}</div>
      `;
      segmentsList.appendChild(row);
    });

    transcriptBox.classList.add('show');
    showToast('success', 'Transcription complete',
      `${words} words · ${lastSegments.length} segments · ${lang}${chunked ? ' · Long file chunked' : ''}`);
  }

  
  // Poll for job status until completed or failed
  function pollStatus(jobId) {
    let pollCount = 0;
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }

    pollTimer = setInterval(async () => {
      pollCount++;
      try {
        const res  = await fetch(`/status/${jobId}`);
        const data = await res.json();

        if (!res.ok && res.status !== 404) {
          clearInterval(pollTimer); pollTimer = null;
          progressWrap.classList.remove('show');
          submitBtn.disabled = false;
          submitBtn.textContent = '🎙 Transcribe File';
          showToast('error', 'Status check failed', data.message || 'Unknown error', false);
          return;
        }

        if (data.status === 'completed') {
          clearInterval(pollTimer); pollTimer = null;
          progressWrap.classList.remove('show');
          submitBtn.disabled = false;
          submitBtn.textContent = '🎙 Transcribe File';

          // Fetch full result
          const resultRes = await fetch(`/result/${jobId}`);
          const resultData = await resultRes.json();
          if (resultRes.ok) {
            displayResult(resultData);
          } else {
            showToast('error', 'Failed to load result', resultData.message || 'Unknown', false);
          }

        } else if (data.status === 'failed') {
          clearInterval(pollTimer); pollTimer = null;
          progressWrap.classList.remove('show');
          submitBtn.disabled = false;
          submitBtn.textContent = '🎙 Transcribe File';
          showToast('error', 'Transcription failed',
            data.error || 'Unknown error', true);  // ← show retry button

        } else {
          // Still processing — keep animating progress
          progressLabel.textContent = `Job ${jobId.slice(0,8)}… · ${steps[Math.min(stepIdx, steps.length-1)]} (waiting)`;
        }
      } catch (err) {
        clearInterval(pollTimer); pollTimer = null;
        progressWrap.classList.remove('show');
        submitBtn.disabled = false;
        submitBtn.textContent = '🎙 Transcribe File';
        showToast('error', 'Connection lost', err.message, false);
      }
    }, 2000); // poll every 2 seconds
  }


  submitBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    submitBtn.disabled = true;
    submitBtn.textContent = '⏳ Transcribing...';
    progressWrap.classList.add('show');
    hideToast(); hideTranscript();

    stepIdx = 0;
    progressLabel.textContent = steps[0];
    stepTimer = setInterval(() => {
      if (stepIdx < steps.length - 1) progressLabel.textContent = steps[++stepIdx];
    }, 4000);

    const fd = new FormData();
    fd.append('audio', selectedFile);

    try {
      const res  = await fetch('/upload', { method: 'POST', body: fd });
      const data = await res.json();
      clearInterval(stepTimer);

      if (res.ok && data.transcript) {
        lastText     = data.transcript;
        lastSegments = data.segments || [];
        const words  = data.transcript.trim().split(/\s+/).length;

        transcriptMeta.innerHTML = `
          <span class="meta-tag">🌐 ${(data.language || 'unknown').toUpperCase()}</span>
          <span class="meta-tag">📝 ${words} words</span>
          <span class="meta-tag">⏱ ${data.duration || '?'}s</span>
          <span class="meta-tag">🔖 ${lastSegments.length} segments</span>
          <span class="meta-tag">${data.chunked ? '🔀 Chunked' : '⚡ Direct'}</span>
          <span class="meta-tag">🤖 Whisper AI</span>
        `;
        transcriptText.textContent = data.transcript.trim();

        segmentsList.innerHTML = '';
        lastSegments.forEach(seg => {
          const row = document.createElement('div');
          row.className = 'seg-row';
          row.innerHTML = `
            <div class="seg-time">
              <span class="seg-start">▶ ${formatTime(seg.start)}</span>
              <span class="seg-end">⏹ ${formatTime(seg.end)}</span>
            </div>
            <div class="seg-text">${seg.text}</div>
          `;
          segmentsList.appendChild(row);
        });

        transcriptBox.classList.add('show');
        showToast('success', 'Transcription complete',
          `${words} words · ${lastSegments.length} segments · ${(data.language||'').toUpperCase()}${data.chunked ? ' · Long file chunked' : ''}`);
      } else {
        showToast('error', 'Transcription failed', data.message || 'Unknown error');
      }
    } catch (err) {
      clearInterval(stepTimer);
      showToast('error', 'Something went wrong', err.message);
    } finally {
      progressWrap.classList.remove('show');
      submitBtn.disabled = false;
      submitBtn.textContent = '🎙 Transcribe File';
    }
  });

  copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(lastText);
    copyBtn.textContent = '✓ Copied!';
    setTimeout(() => copyBtn.textContent = '📋 Copy', 2000);
  });

  dlBtn.addEventListener('click', () => {
    let content = lastText.trim() + '\n\n--- SEGMENTS ---\n';
    lastSegments.forEach(s => {
      content += `[${formatTime(s.start)} --> ${formatTime(s.end)}]  ${s.text}\n`;
    });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([content], {type:'text/plain'}));
    a.download = 'transcript.txt'; a.click();
  });

  dlJsonBtn.addEventListener('click', () => {
    const payload = { transcript: lastText.trim(), segments: lastSegments };
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type:'application/json'}));
    a.download = 'transcript.json'; a.click();
  });
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return jsonify({'message': 'No file provided.'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'message': 'No file selected.'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        return jsonify({'message': f'"{ext}" is not supported. Use WAV, MP3, M4A, OGG or FLAC.'}), 400

    # ── CONCURRENT UPLOAD HANDLING ──
    # Each request gets its own isolated folder (UUID) so multiple
    # users uploading at the same time never touch each other's files
    request_dir = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()))
    os.makedirs(request_dir, exist_ok=True)

    tmp_path = os.path.join(request_dir, "audio" + ext)
    wav_path = os.path.join(request_dir, "converted.wav")
    file.save(tmp_path)

    try:
        # ── FORMAT HANDLING ──
        # pydub + FFmpeg converts any format to 16kHz mono WAV
        AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
        AudioSegment.ffprobe   = r"C:\ffmpeg\bin\ffprobe.exe"

        audio = AudioSegment.from_file(tmp_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")

        duration_ms = len(audio)
        chunked     = duration_ms > CHUNK_THRESHOLD_MS

        # ── LONG FILE HANDLING ──
        # Files > 10 min are split into 5-min overlapping chunks
        # Short files are transcribed directly — no unnecessary overhead
        if chunked:
            print(f"Long file detected ({duration_ms/60000:.1f} min) — chunking")
            result = transcribe_in_chunks(audio, request_dir)
            segments = result['segments']
            language = result['language']
            text     = result['text']
            duration = round(duration_ms / 1000, 1)
        else:
            print(f"Short file ({duration_ms/60000:.1f} min) — direct transcription")
            result   = model.transcribe(wav_path, verbose=False)
            language = result.get('language', 'unknown')
            text     = result['text']
            duration = round(result['segments'][-1]['end'], 1) if result['segments'] else 0
            segments = [
                {
                    'id':    i,
                    'start': round(seg['start'], 2),
                    'end':   round(seg['end'],   2),
                    'text':  seg['text'].strip()
                }
                for i, seg in enumerate(result['segments'])
            ]

        return jsonify({
            'message':    'File processed.',
            'transcript': text,
            'language':   language,
            'duration':   duration,
            'segments':   segments,
            'chunked':    chunked
        })

    except Exception as e:
        return jsonify({'message': f'Error during transcription: {str(e)}'}), 500

    finally:
        # ── CLEANUP ──
        # Delete entire request folder — handles all temp files in one shot
        # Works safely even if some files failed to create
        shutil.rmtree(request_dir, ignore_errors=True)


if __name__ == '__main__':
    app.run(debug=True, port=5000)