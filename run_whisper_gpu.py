"""
Version GPU de la transcription Whisper.

Caractéristiques:
- Force l'utilisation du GPU (CUDA)
- Modèle par défaut: medium
- Langue par défaut: en
- Liste et log les fichiers du dossier `audio_ref`, transcrit et vérifie une phrase

Pré-requis (dans la venv):
  python -m pip uninstall -y torch torchvision torchaudio
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio -U
  python -m pip install -U openai-whisper

Exécution:
  python run_whisper_gpu.py --phrase "Appeler Matt" --verbose

Remarque:
- Si CUDA n'est pas disponible, le script s'arrête avec un message d'erreur.
- Si vous avez des soucis d'installation, essayez cu121/cu124 à la place de cu128.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional


AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".webm",
    ".aac",
    ".wma",
    ".opus",
    ".mkv",
    ".mp4",
}


def setup_logging(log_file: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("whisper_gpu_runner")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    except Exception:
        logger.warning("Impossible d'initialiser le fichier de logs, poursuite en console uniquement.")

    return logger


def check_ffmpeg(logger: logging.Logger) -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("FFmpeg introuvable. Installez-le et mettez-le dans le PATH.")
        return False


def find_audio_files(root_dir: Path) -> List[Path]:
    files: List[Path] = []
    if not root_dir.exists():
        return files
    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)
    return sorted(files)


def normalize_text(text: str) -> str:
    text_nfkd = unicodedata.normalize("NFKD", text)
    text_no_accents = "".join(ch for ch in text_nfkd if not unicodedata.combining(ch))
    return text_no_accents.lower()


def contains_phrase(transcript: str, phrase: str) -> bool:
    return normalize_text(phrase) in normalize_text(transcript)


def load_whisper_model_gpu(model_name: str, logger: logging.Logger):
    try:
        import whisper  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        logger.error("Dépendances manquantes. Installez PyTorch CUDA et openai-whisper dans la venv.")
        raise SystemExit(1) from exc

    if not torch.cuda.is_available():
        logger.error(
            "CUDA non disponible. Vérifiez votre installation GPU de PyTorch (cu1281) et vos pilotes NVIDIA."
        )
        logger.error(
            "Exemples:\n"
            "  python -m pip uninstall -y torch torchvision torchaudio\n"
            "  python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio -U"
        )
        raise SystemExit(1)

    device_name = torch.cuda.get_device_name(0)
    # fp16 recommandé sur GPU moderne; forcer le dossier local `model/`
    model_dir = Path(__file__).resolve().parent / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Chargement du modèle Whisper '{model_name}' sur cuda (GPU: {device_name}) (download_root={model_dir}) ..."
    )
    model = whisper.load_model(model_name, device="cuda", download_root=str(model_dir))
    use_fp16 = True
    return model, use_fp16


def transcribe_file(
    model,
    file_path: Path,
    language: Optional[str],
    use_fp16: bool,
    logger: logging.Logger,
) -> str:
    start_time = time.time()
    logger.info(f"Transcription: {file_path}")

    try:
        result = model.transcribe(
            str(file_path),
            language=language,
            task="transcribe",
            fp16=use_fp16,
            verbose=False,
        )
    except Exception as exc:
        logger.error(f"Échec de la transcription pour {file_path}: {exc}")
        return ""

    text = (result.get("text") or "").strip()
    elapsed = time.time() - start_time
    logger.info(f"Terminé en {elapsed:.1f}s | Longueur texte: {len(text)} caractères")
    return text


def main(argv: Optional[Iterable[str]] = None) -> int:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Transcription Whisper (GPU/CUDA) avec vérification d'une phrase.")
    parser.add_argument(
        "--dir",
        dest="audio_dir",
        type=str,
        default=str(script_dir / "audio_ref"),
        help="Dossier contenant les fichiers audio (défaut: ./audio_ref)",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        type=str,
        default=os.environ.get("WHISPER_MODEL", "medium"),
        help="Nom du modèle Whisper (tiny/base/small/medium/large). Défaut: medium",
    )
    parser.add_argument(
        "--language",
        dest="language",
        type=str,
        default=os.environ.get("WHISPER_LANGUAGE", "en"),
        help="Langue attendue (ex: en, fr). Défaut: en",
    )
    parser.add_argument(
        "--phrase",
        dest="phrase",
        type=str,
        default=os.environ.get("TARGET_PHRASE", "Appeler Matt"),
        help="Phrase à vérifier dans la transcription",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Activer les logs détaillés",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    log_file = script_dir / "whisper_gpu.log"
    logger = setup_logging(log_file=log_file, verbose=args.verbose)

    audio_dir = Path(args.audio_dir)
    logger.info(f"Dossier audio: {audio_dir}")

    ffmpeg_ok = check_ffmpeg(logger)
    if not ffmpeg_ok:
        logger.warning("FFmpeg manquant : la transcription risque d'échouer tant que FFmpeg n'est pas disponible.")

    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        logger.info("Aucun fichier audio trouvé. Ajoutez des fichiers dans 'audio_ref' puis relancez.")
        return 0

    logger.info("Fichiers détectés:")
    for p in audio_files:
        logger.info(f" - {p}")

    model, use_fp16 = load_whisper_model_gpu(args.model_name, logger)

    logger.info("Début des transcriptions (GPU)...")
    found_count = 0
    total = 0
    for file_path in audio_files:
        total += 1
        text = transcribe_file(model, file_path, args.language, use_fp16, logger)
        if not text:
            logger.info(f"[{file_path.name}] Aucun texte obtenu.")
            continue

        # Afficher et logger la transcription complète
        logger.info(f"Transcription complète ({file_path.name}):\n{text}")

        present = contains_phrase(text, args.phrase)
        if present:
            found_count += 1
            logger.info(f"[OK] La phrase cible est présente dans '{file_path.name}'.")
        else:
            logger.info(f"[NON TROUVÉ] La phrase n'est pas présente dans '{file_path.name}'.")

        snippet = text[:300].replace("\n", " ") + ("..." if len(text) > 300 else "")
        logger.debug(f"Extrait transcription ({file_path.name}): {snippet}")

    logger.info(f"Terminé. Fichiers traités: {total} | Phrase trouvée dans: {found_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


