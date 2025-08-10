"""
Script de transcription audio avec Whisper.

Fonctions principales:
- Lister et logger tous les fichiers audio dans le dossier `audio_ref`
- Transcrire chaque fichier avec Whisper
- Vérifier si la transcription contient une phrase cible (par défaut: "Appeler Matt")

Utilisation rapide (Windows / PowerShell):
  1) Installer les dépendances Python:
     python -m pip install -U openai-whisper

  2) Installer FFmpeg (requis pour le décodage audio), par exemple avec winget:
     winget install Gyan.FFmpeg
     (ou téléchargez depuis https://www.gyan.dev/ffmpeg/builds/ et ajoutez ffmpeg.exe au PATH)

  3) Placez vos fichiers audio dans le dossier `audio_ref/`

  4) Lancer:
     python run_whisper.py --phrase "Appeler Matt"

Options:
  --dir       Dossier contenant les audios (défaut: audio_ref)
  --model     Modèle Whisper (tiny, base, small, medium, large; défaut: small)
  --language  Langue attendue (ex: fr, en). Par défaut: fr
  --phrase    Phrase à vérifier dans la transcription
  --verbose   Logs détaillés

Notes:
- Sur CPU, préférez un modèle plus petit (tiny/base/small) pour de meilleures performances.
- Le script crée un fichier de logs `whisper.log` à côté de ce script.
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
    logger = logging.getLogger("whisper_runner")
    logger.setLevel(logging.DEBUG)

    # Éviter les handlers dupliqués si relancé dans le même process
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
        # Si on ne peut pas écrire le fichier de logs, continuer avec la console uniquement
        logger.warning("Impossible d'initialiser le fichier de logs, poursuite en console uniquement.")

    return logger


def check_ffmpeg(logger: logging.Logger) -> bool:
    """Vérifie la disponibilité de ffmpeg dans le PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning(
            "FFmpeg introuvable. Installez-le (ex: `winget install Gyan.FFmpeg`) et assurez-vous qu'il est dans le PATH."
        )
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
    """Normalise pour comparaison insensible aux accents/majuscules."""
    text_nfkd = unicodedata.normalize("NFKD", text)
    text_no_accents = "".join(ch for ch in text_nfkd if not unicodedata.combining(ch))
    return text_no_accents.lower()


def contains_phrase(transcript: str, phrase: str) -> bool:
    return normalize_text(phrase) in normalize_text(transcript)


def load_whisper_model(model_name: str, logger: logging.Logger):
    try:
        import whisper  # type: ignore
    except ImportError as exc:
        logger.error("Le paquet 'openai-whisper' n'est pas installé. Exécutez: python -m pip install -U openai-whisper")
        raise SystemExit(1) from exc

    device = "cpu"
    fp16 = False
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            device = "cuda"
            fp16 = True
    except Exception:
        # torch absent ou problème de détection, rester en CPU
        pass

    # Dossier local pour stocker les modèles
    model_dir = Path(__file__).resolve().parent / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Chargement du modèle Whisper '{model_name}' sur {device} (download_root={model_dir}) ...")
    model = whisper.load_model(model_name, device=device, download_root=str(model_dir))
    return model, fp16


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

    parser = argparse.ArgumentParser(description="Transcription batch avec Whisper et vérification d'une phrase.")
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
        default=os.environ.get("WHISPER_MODEL", "small"),
        help="Nom du modèle Whisper (tiny/base/small/medium/large). Défaut: small",
    )
    parser.add_argument(
        "--language",
        dest="language",
        type=str,
        default=os.environ.get("WHISPER_LANGUAGE", "fr"),
        help="Langue attendue (ex: fr, en). Défaut: fr",
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

    log_file = script_dir / "whisper.log"
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

    model, use_fp16 = load_whisper_model(args.model_name, logger)

    logger.info("Début des transcriptions...")
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

        # Afficher un extrait de la transcription pour vérification rapide
        snippet = text[:300].replace("\n", " ") + ("..." if len(text) > 300 else "")
        logger.debug(f"Extrait transcription ({file_path.name}): {snippet}")

    logger.info(f"Terminé. Fichiers traités: {total} | Phrase trouvée dans: {found_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


