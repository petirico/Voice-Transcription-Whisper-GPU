## Whisper – Batch transcription (CPU & GPU)

Minimal project to transcribe audio files from the `audio_ref/` folder to text using Whisper (OpenAI open source), with two scripts:
- `run_whisper.py` (CPU)
- `run_whisper_gpu.py` (GPU CUDA)

Transcriptions and status are printed to the terminal and logged to files (`whisper.log`, `whisper_gpu.log`). Models are downloaded/stored under the project `model/` folder.

### Useful links
- Whisper (OpenAI, open source): `https://github.com/openai/whisper`
- PyTorch CUDA 12.8 (cu128) index: `https://download.pytorch.org/whl/cu128`
- FFmpeg for Windows (Gyan builds): `https://www.gyan.dev/ffmpeg/builds/`

## Prerequisites
- Windows 10/11 (PowerShell) with Python 3.10+ installed and on PATH
- FFmpeg available on PATH (audio decoding)
- Optional (GPU): NVIDIA GPU (e.g., RTX 5080) + up-to-date NVIDIA drivers + CUDA runtime via PyTorch cu128

## Quick install (CPU)
```powershell
cd D:\TTS\Whisper
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
python -m pip install -U openai-whisper
winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
```

## Installation (GPU – CUDA)
Install PyTorch with CUDA 12.8 (cu128), then Whisper:
```powershell
cd D:\TTS\Whisper
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio -U
python -m pip install -U openai-whisper
python - << "PY"
import torch
print('cuda available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
```

## Whisper models (local `model/`)
- Weights are downloaded automatically on first run into `model/`.
- You can also pre-download:
```powershell
.\.venv\Scripts\Activate.ps1
python - << "PY"
import whisper
from pathlib import Path
Path('model').mkdir(exist_ok=True)
for name in ['tiny','base','small','medium']:
    whisper.load_model(name, download_root='model')
print('OK')
PY
```

## Usage (CPU)
```powershell
.\.venv\Scripts\Activate.ps1
python run_whisper.py --phrase "Appeler Matt" --model small --language fr --verbose
```

## Usage (GPU)
```powershell
.\.venv\Scripts\Activate.ps1
python run_whisper_gpu.py --phrase "Appeler Matt" --model medium --language en --verbose
```

## What the scripts do
- List all audio files in `audio_ref/` (common formats: wav, mp3, m4a, flac, ogg, webm, aac, wma, opus, mkv, mp4)
- Transcribe each file with Whisper (configurable model and language)
- Print and log the full transcription to the terminal and log files
- Check if a target phrase (default "Appeler Matt") is present (accent/case-insensitive)
- Final summary: number of files and matches

## Main CLI options
- `--dir`: audio folder (default: `audio_ref`)
- `--model`: `tiny | base | small | medium | large` (default: `small` for CPU, `medium` for GPU)
- `--language`: Whisper language code (e.g., `fr`, `en`) – CPU default: `fr`, GPU default: `en`
- `--phrase`: phrase to search in the transcription
- `--verbose`: detailed logs

## Output & logs
- Terminal: progress, full transcription per file, and phrase search result
- Log files: `whisper.log` (CPU) and `whisper_gpu.log` (GPU)
- Models: `model/` (e.g., `tiny.pt`, `base.pt`, `medium.pt`)

## Structure
```
Whisper/
  audio_ref/            # Put your audio files here
  model/                # Local model weights (.pt)
  run_whisper.py        # CPU version
  run_whisper_gpu.py    # GPU (CUDA) version
  whisper.log           # CPU logs
  whisper_gpu.log       # GPU logs
  .venv/                # Virtual environment (git-ignored)
  .gitignore
  README.md
```

## Troubleshooting
- `AttributeError: module 'whisper' has no attribute 'load_model'`: do not name your script `whisper.py` (name clash). Use `run_whisper*.py`.
- `FFmpeg not found`: install FFmpeg and add it to PATH (see link above).
- `CUDA not available`: check PyTorch cu128 install, NVIDIA drivers, and that `torch.cuda.is_available()` is `True`.

## Publish to GitHub
`.gitignore` is included to ignore `.venv/` and all `*.pt` model files. Initialize a repo, commit, and push:
```powershell
git init
git add .
git commit -m "Init Whisper CPU/GPU + scripts + README"
git branch -M main
git remote add origin <REPO_URL>
git push -u origin main
```


