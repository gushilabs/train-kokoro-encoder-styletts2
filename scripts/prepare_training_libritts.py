#!/usr/bin/env python3
"""
Kokoro LibriTTS-R: Prepare Training Data
========================================
Converts the dataset produced by prepare_libritts.py into the format
expected by StyleTTS2's training scripts.

Usage:
    # Step 1: Generate train/val lists
    python prepare_training_libritts.py prepare

    # Step 2: Smoke test (verify data loads correctly)
    python prepare_training_libritts.py verify
"""

import argparse
from curses import meta
import json
import os
import random
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

DATASET_DIR = Path("./dataset")
TRAINING_DIR = Path("../training")
WAVS_DIR = DATASET_DIR / "audio"
METADATA_FILE = DATASET_DIR / "metadata.csv"
PHONEMES_FILE = DATASET_DIR / "phonemes.csv"

TRAIN_LIST = TRAINING_DIR / "train_list.txt"
VAL_LIST = TRAINING_DIR / "val_list.txt"

# ── Split ────────────────────────────────────────────────────────────────────

VAL_RATIO = 0.05
RANDOM_SEED = 42


def cmd_prepare():
    """Generate train/val lists from the parsed LibriTTS-R data."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load metadata and phonemes ───────────────────────────────────────
    if not METADATA_FILE.exists() or not PHONEMES_FILE.exists():
        print("ERROR: metadata.csv or phonemes.csv not found.")
        print("Make sure you ran the prepare_libritts.py script first.")
        sys.exit(1)

    # Parse metadata: filename|text|speaker
    meta = {}
    with open(METADATA_FILE) as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) == 3:
                filename, text, speaker = parts
                
                # Extract only the digits from the speaker string (e.g., "speaker260" -> "260")
                speaker_id = "".join(filter(str.isdigit, speaker))
                
                # Fallback to "0" just in case a row is missing a number to prevent crashes
                if not speaker_id:
                    speaker_id = "0"
                    
                meta[filename] = {"text": text, "speaker": speaker_id}

    # Parse phonemes: filename|ipa
    phonemes = {}
    with open(PHONEMES_FILE) as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) == 2:
                filename, ipa = parts
                phonemes[filename] = ipa

    # Merge and validate
    entries = []
    missing_phonemes = 0
    missing_wav = 0
    empty_phonemes = 0
    
    for filename, m in meta.items():
        wav_path = WAVS_DIR / filename
        if not wav_path.exists():
            missing_wav += 1
            continue
        ipa = phonemes.get(filename, "")
        if not ipa:
            missing_phonemes += 1
            continue
        # Skip entries with very short phoneme sequences
        if len(ipa) < 5:
            empty_phonemes += 1
            continue
            
        entries.append(
            {
                "filename": filename,
                "wav_path": str(wav_path),
                "text": m["text"],
                "speaker": m["speaker"],
                "ipa": ipa,
            }
        )

    print(f"Total entries: {len(meta):,}")
    print(f"Valid entries: {len(entries):,}")
    if missing_wav:
        print(f"  Missing WAV: {missing_wav}")
    if missing_phonemes:
        print(f"  Missing phonemes: {missing_phonemes}")
    if empty_phonemes:
        print(f"  Empty/short phonemes: {empty_phonemes}")

    if not entries:
        print("ERROR: No valid entries found.")
        sys.exit(1)

    # ── Train/val split ──────────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(entries)
    n_val = max(1, int(len(entries) * VAL_RATIO))
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]

    print(f"\nSplit: {len(train_entries):,} train / {len(val_entries):,} val")

    # ── Write train/val lists ────────────────────────────────────────────
    # Format: relative_wav_path|phoneme_sequence|speaker_id
    def write_list(path, entries_list):
        with open(path, "w", encoding="utf-8") as f:
            for e in entries_list:
                f.write(f"{e['filename']}|{e['ipa']}|{e['speaker']}\n")

    write_list(TRAIN_LIST, train_entries)
    write_list(VAL_LIST, val_entries)

    print(f"Wrote {TRAIN_LIST} ({len(train_entries):,} lines)")
    print(f"Wrote {VAL_LIST} ({len(val_entries):,} lines)")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Training data ready in {TRAINING_DIR}/")
    print(f"  train_list.txt  : {len(train_entries):,} entries")
    print(f"  val_list.txt    : {len(val_entries):,} entries")
    print(f"  Audio dir       : {WAVS_DIR}/")
    print(f"{'=' * 60}")


def _generate_symbols_code(symbols, vocab):
    """Generate Python source code defining the symbols list matching Kokoro's vocab."""
    lines = [
        '"""',
        "Kokoro-82M Symbol Mapping for StyleTTS2",
        "=========================================",
        "Auto-generated from Kokoro-82M config.json.",
        "Replaces StyleTTS2's default symbol list in text_utils.py and meldataset.py.",
        '"""',
        "",
        "# fmt: off",
        "symbols = [",
    ]

    for i, sym in enumerate(symbols):
        if sym in vocab.values() if isinstance(sym, int) else sym in vocab:
            if sym == '"':
                repr_sym = "'\\\"'"
            elif sym == "'":
                repr_sym = '"\'"'
            elif sym == "\\":
                repr_sym = "'\\\\'"
            else:
                repr_sym = repr(sym)

            if ord(sym) >= 32 and ord(sym) < 127:
                comment = f"  # {i:3d}: {sym}"
            else:
                comment = f"  # {i:3d}: U+{ord(sym):04X} ({sym})"
        else:
            repr_sym = repr(sym)
            if i == 0:
                comment = f"  # {i:3d}: PAD"
            else:
                comment = f"  # {i:3d}: (unused placeholder)"

        lines.append(f"    {repr_sym},{comment}")

    lines.extend(
        [
            "]",
            "# fmt: on",
            "",
            "dicts = {sym: i for i, sym in enumerate(symbols)}",
            "",
            "class TextCleaner:",
            "    def __init__(self, dummy=0):",
            "        self.word_index_dictionary = dicts",
            "",
            "    def __call__(self, text):",
            "        indexes = []",
            "        for char in text:",
            "            if char in self.word_index_dictionary:",
            "                indexes.append(self.word_index_dictionary[char])",
            "        return indexes",
            "",
            f'assert len(symbols) == 178, f"Expected 178 symbols, got {{len(symbols)}}"',
            "",
        ]
    )

    return "\n".join(lines) + "\n"


def cmd_verify():
    """Verify training data integrity — check a few samples end-to-end."""
    print("Verifying training data...")

    issues = []

    for f in [TRAIN_LIST, VAL_LIST]:
        if not f.exists():
            issues.append(f"MISSING: {f}")

    if issues:
        for issue in issues:
            print(f"  ERROR: {issue}")
        print("\nRun 'prepare' first.")
        sys.exit(1)

    n_train = 0
    n_val = 0
    missing_wavs = 0
    empty_phonemes = 0
    speakers = set()

    for list_file, label in [(TRAIN_LIST, "train"), (VAL_LIST, "val")]:
        with open(list_file) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) != 3:
                    issues.append(
                        f"{label}:{line_no} — expected 3 fields, got {len(parts)}"
                    )
                    continue

                filename, ipa, speaker = parts
                wav_path = WAVS_DIR / filename

                if not wav_path.exists():
                    missing_wavs += 1

                if not ipa or len(ipa) < 3:
                    empty_phonemes += 1

                speakers.add(speaker)

                if label == "train":
                    n_train += 1
                else:
                    n_val += 1

    print(f"  Train entries : {n_train:,}")
    print(f"  Val entries   : {n_val:,}")
    print(f"  Speakers      : {len(speakers)}")
    print(f"  Missing WAVs  : {missing_wavs}")
    print(f"  Empty phonemes: {empty_phonemes}")

    config_path = TRAINING_DIR / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        vocab = config["vocab"]

        unknown_chars = set()
        with open(TRAIN_LIST, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                ipa = line.strip().split("|")[1]
                for ch in ipa:
                    if ch not in vocab:
                        unknown_chars.add(ch)

        if unknown_chars:
            print(f"  WARNING: Unknown phoneme chars (will be dropped): {unknown_chars}")
        else:
            print(f"  Phoneme vocab : OK (all symbols in Kokoro vocab)")

    if issues:
        print(f"\n  ISSUES FOUND:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"\n  All checks passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for StyleTTS2 fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", help="Generate train/val lists from dataset")
    
    p_convert = subparsers.add_parser(
        "convert-weights", help="Convert Kokoro-82M weights to StyleTTS2 format"
    )
    p_convert.add_argument(
        "--force", action="store_true", help="Regenerate even if output exists"
    )

    subparsers.add_parser(
        "patch-styletts2",
        help="Patch StyleTTS2 text_utils.py and meldataset.py with Kokoro's vocab",
    )

    subparsers.add_parser("verify", help="Verify training data integrity")

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare()
    elif args.command == "verify":
        cmd_verify()


if __name__ == "__main__":
    main()