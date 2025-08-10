## Whisper – Batch transcription (CPU & GPU)

Projet minimal pour transcrire des fichiers audio d'un dossier `audio_ref/` en texte avec Whisper (open source d'OpenAI), avec deux scripts:
- `run_whisper.py` (CPU)
- `run_whisper_gpu.py` (GPU CUDA)

Les transcriptions et le statut sont affichés dans le terminal et journalisés dans des fichiers de logs (`whisper.log`, `whisper_gpu.log`). Les modèles sont téléchargés/localisés dans le dossier `model/` du projet.

### Liens utiles
- Whisper (open source, OpenAI): `https://github.com/openai/whisper`
- PyTorch index CUDA 12.8 (cu128): `https://download.pytorch.org/whl/cu128`
- FFmpeg Windows (builds Gyan): `https://www.gyan.dev/ffmpeg/builds/`

## Prérequis
- Windows 10/11 (PowerShell) avec Python 3.10+ installé (et ajouté au PATH)
- FFmpeg disponible dans le PATH (décodage audio)
- Optionnel (GPU): Carte NVIDIA (ex: RTX 5080) + pilotes NVIDIA à jour + CUDA runtime via PyTorch cu128

## Installation rapide (CPU)
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
Installe PyTorch avec CUDA 12.8 (cu128), puis Whisper:
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

## Modèles Whisper (local `model/`)
- Les poids sont téléchargés automatiquement au premier lancement dans `model/`.
- Vous pouvez aussi pré-télécharger:
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

## Utilisation (CPU)
```powershell
.\.venv\Scripts\Activate.ps1
python run_whisper.py --phrase "Appeler Matt" --model small --language fr --verbose
```

## Utilisation (GPU)
```powershell
.\.venv\Scripts\Activate.ps1
python run_whisper_gpu.py --phrase "Appeler Matt" --model medium --language en --verbose
```

## Ce que font les scripts
- Lister tous les fichiers audio du dossier `audio_ref/` (formats courants: wav, mp3, m4a, flac, ogg, webm, aac, wma, opus, mkv, mp4)
- Transcrire chaque fichier avec Whisper (modèle et langue configurables)
- Afficher et logger la transcription complète dans le terminal et les fichiers de logs
- Vérifier si une phrase cible (par défaut "Appeler Matt") est présente dans la transcription (comparaison insensible aux accents/majuscules)
- Résumé final: nombre de fichiers et de correspondances

## Options principales (CLI)
- `--dir`: dossier des audios (défaut: `audio_ref`)
- `--model`: `tiny | base | small | medium | large` (défaut: `small` pour CPU, `medium` pour GPU)
- `--language`: code langue Whisper (ex: `fr`, `en`) – défaut CPU: `fr`, GPU: `en`
- `--phrase`: phrase à rechercher dans la transcription
- `--verbose`: logs détaillés

## Sortie & logs
- Terminal: progression, transcription complète par fichier, et résultat de la recherche de phrase
- Fichiers de logs: `whisper.log` (CPU) et `whisper_gpu.log` (GPU)
- Modèles: `model/` (ex: `tiny.pt`, `base.pt`, `medium.pt`)

## Structure
```
Whisper/
  audio_ref/            # Déposez ici vos fichiers audio
  model/                # Poids des modèles (.pt) téléchargés localement
  run_whisper.py        # Version CPU
  run_whisper_gpu.py    # Version GPU (CUDA)
  whisper.log           # Logs CPU
  whisper_gpu.log       # Logs GPU
  .venv/                # Environnement virtuel (ignoré par git)
  .gitignore
  README.md
```

## Dépannage rapide
- `AttributeError: module 'whisper' has no attribute 'load_model'`: ne nommez pas votre script `whisper.py` (conflit de nom). Utilisez `run_whisper*.py`.
- `FFmpeg introuvable`: installez FFmpeg et ajoutez-le au PATH (voir lien ci-dessus).
- `CUDA non disponible`: vérifiez l’installation PyTorch cu128, vos pilotes NVIDIA et que `torch.cuda.is_available()` renvoie `True`.

## Publication sur GitHub
Un `.gitignore` est inclus pour ignorer `.venv/` et tous les fichiers `*.pt` (modèles). Commencez un dépôt, validez et poussez:
```powershell
git init
git add .
git commit -m "Init Whisper CPU/GPU + scripts + README"
git branch -M main
git remote add origin <URL_DU_DEPOT>
git push -u origin main
```


