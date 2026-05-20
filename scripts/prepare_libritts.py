#!/usr/bin/env python3
"""
Kokoro LibriTTS-R Training Dataset Pipeline
===========================================
Processes the pre-transcribed LibriTTS-R dataset into Kokoro TTS format.

Usage:
    # Step 1: Scan LibriTTS-R, read text, and filter
    python prepare_libritts.py scan /path/to/LibriTTS_R/train-clean-100

    # Step 2: Convert audio + generate IPA + write final dataset
    python prepare_libritts.py format
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────

DATASET_DIR = Path("./dataset")
AUDIO_DIR = DATASET_DIR / "audio"
SCANNED_FILE = DATASET_DIR / "scanned.jsonl"
METADATA_FILE = DATASET_DIR / "metadata.csv"
PHONEMES_FILE = DATASET_DIR / "phonemes.csv"
STATS_FILE = DATASET_DIR / "stats.json"

# ── Filtering thresholds ─────────────────────────────────────────────────────

MIN_DURATION_S = 1.0
MAX_DURATION_S = 10.0
MIN_WORDS = 2

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Scan and Filter
# ─────────────────────────────────────────────────────────────────────────────

def _get_duration(path: Path) -> float:
    """Fast duration extraction via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0

def cmd_scan(libritts_dir: str):
    """Scan LibriTTS-R directory, read text, filter, and save to jsonl."""
    base_dir = Path(libritts_dir)
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        sys.exit(1)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {base_dir} for WAV files...")
    wav_files = list(base_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} audio files. Processing text and filtering...")

    kept = []
    reasons = {"missing_text": 0, "too_short": 0, "too_long": 0, "too_few_words": 0}

    with open(SCANNED_FILE, "w", encoding="utf-8") as out:
        for wav_path in tqdm(wav_files, desc="Scanning"):
            # LibriTTS-R text files match the wav name but end in .normalized.txt
            text_path = wav_path.with_suffix(".normalized.txt")
            
            if not text_path.exists():
                reasons["missing_text"] += 1
                continue

            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if len(text.split()) < MIN_WORDS:
                reasons["too_few_words"] += 1
                continue

            duration = _get_duration(wav_path)
            if duration < MIN_DURATION_S:
                reasons["too_short"] += 1
                continue
            if duration > MAX_DURATION_S:
                reasons["too_long"] += 1
                continue

            # In LibriTTS, the speaker ID is the top-level folder name (e.g., '198')
            # path parts: [... , 'train-clean-100', '198', '129977', '198_129977...wav']
            speaker_id = f"speaker_{wav_path.parent.parent.name}"

            entry = {
                "hash": wav_path.stem,
                "original_path": str(wav_path),
                "speaker": speaker_id,
                "text": text,
                "duration": round(duration, 3)
            }
            kept.append(entry)
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nScan complete. Kept {len(kept):,} valid files.")
    print("Filtered out:")
    for reason, count in reasons.items():
        if count > 0:
            print(f"  {reason}: {count}")
    print(f"Saved manifest to {SCANNED_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Format (Convert & G2P)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_format():
    """Convert WAVs to 24kHz mono, generate English IPA, write final dataset."""
    if not SCANNED_FILE.exists():
        print("No scanned.jsonl found. Run 'scan' first.")
        sys.exit(1)

    entries = [json.loads(line) for line in open(SCANNED_FILE, encoding="utf-8")]
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Create speaker directories
    speakers = set(e["speaker"] for e in entries)
    for spk in speakers:
        (AUDIO_DIR / spk).mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(entries):,} files to 24kHz Mono WAV...")
    errors = 0
    converted = 0

    for entry in tqdm(entries, desc="Audio Processing"):
        spk = entry["speaker"]
        wav_path = AUDIO_DIR / spk / f"{entry['hash']}.wav"
        entry["final_wav_path"] = str(wav_path)

        if wav_path.exists():
            continue # Skip if already converted

        result = subprocess.run([
            "ffmpeg", "-y", "-i", entry["original_path"],
            "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", str(wav_path)
        ], capture_output=True)
        
        if result.returncode != 0:
            errors += 1
        else:
            converted += 1

    print(f"Audio conversion done. Converted: {converted} | Errors: {errors}")

    print("Generating English IPA phonemes via misaki...")
    try:
        from misaki import espeak
        # Kokoro expects American English ('en-us') phonemes
        g2p = espeak.EspeakG2P(language="en-us")
    except ImportError:
        print("Error: 'misaki' package not found. Run: pip install misaki[en]")
        sys.exit(1)

    metadata_rows = []
    phoneme_rows = []
    ipa_errors = 0

    for entry in tqdm(entries, desc="G2P (Phonemes)"):
        text = entry["text"]
        spk = entry["speaker"]
        wav_name = f"{spk}/{entry['hash']}.wav"

        try:
            phonemes, _ = g2p(text)
        except Exception:
            ipa_errors += 1
            continue

        metadata_rows.append(f"{wav_name}|{text}|{spk}")
        phoneme_rows.append(f"{wav_name}|{phonemes}")

    if ipa_errors:
        print(f"IPA generation errors: {ipa_errors}")

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        f.write("filename|text|speaker\n")
        f.write("\n".join(metadata_rows) + "\n")

    with open(PHONEMES_FILE, "w", encoding="utf-8") as f:
        f.write("filename|ipa\n")
        f.write("\n".join(phoneme_rows) + "\n")

    total_duration = sum(e["duration"] for e in entries)
    print(f"\nDataset perfectly formatted for Kokoro!")
    print(f"  Total Files: {len(metadata_rows):,}")
    print(f"  Total Hours: {total_duration / 3600:.2f}h")
    print(f"  Speakers   : {len(speakers)}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LibriTTS-R Pipeline for Kokoro")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_scan = subparsers.add_parser("scan", help="Scan and filter LibriTTS-R directory")
    p_scan.add_argument("libritts_dir", help="Path to the train-clean-100 folder")

    p_format = subparsers.add_parser("format", help="Convert audio and generate IPA")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args.libritts_dir)
    elif args.command == "format":
        cmd_format()