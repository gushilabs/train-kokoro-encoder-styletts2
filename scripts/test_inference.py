#!/usr/bin/env python3
"""
Kokoro: Test Inference
==============================
Tests the fine-tuned Kokoro model with a  phonetic test set.

Pre-requisites:
    pip install -q kokoro>=0.9.4 soundfile
    apt-get -qq -y install espeak-ng > /dev/null 2>&1

Usage:
    python scripts/test_inference.py \
        --model ~/Downloads/models/kokoro-v1_0/kokoro-v1_0.pth \
        --voicepack voices/am_1epoch34_2epoch6_speaker5703.pt \
        --output-dir test_output/epoch6

"""

import argparse
import sys
from pathlib import Path

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Printing, in the only sense with which we are at present concerned, differs from most arts.",
    "Why did you do that? That is absolutely incredible!",
    "The total cost is exactly one hundred twenty-three million dollars.",
    "She sells seashells by the seashore.",
    "Are you serious right now? I can't believe it!",
    "A swift red fox gracefully cleared the sleeping dog."
]

def run_inference(
    model_path: str,
    voicepack_path: str,
    output_dir: str,
    device: str = "auto",
):
    """Run inference on the test set."""
    import torch
    import soundfile as sf
    from kokoro import KModel, KPipeline

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model with our fine-tuned weights and config
    print(f"Loading model from: {model_path}")
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", model=model_path)
    kmodel = kmodel.to(device).eval()

    # Create pipeline with lang_code
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=kmodel)

    # Load voicepack
    print(f"Loading voicepack: {voicepack_path}")
    voice = torch.load(voicepack_path, map_location="cpu", weights_only=True)

    # Create output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate audio for each test sentence
    print(f"\nGenerating {len(TEST_SENTENCES)} test sentences...\n")
    for i, text in enumerate(TEST_SENTENCES):
        print(f"[{i + 1}/{len(TEST_SENTENCES)}] {text[:60]}...")
        try:
            generator = pipeline(text, voice=voice, speed=1)
            all_audio = []
            for gs, ps, audio in generator:
                print(f"  phonemes: {ps[:60]}...")
                all_audio.append(audio)

            if all_audio:
                import numpy as np

                combined = np.concatenate(all_audio)
                wav_path = out / f"test_{i + 1:02d}.wav"
                sf.write(str(wav_path), combined, 24000)
                duration = len(combined) / 24000
                print(f"  saved: {wav_path} ({duration:.1f}s)")
            else:
                print(f"  WARNING: No audio generated")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone! Test audio saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Test fine-tuned Trained Kokoro model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        help="Path to already-converted Kokoro-format weights (.pth)",
    )
    parser.add_argument(
        "--voicepack",
        required=True,
        help="Path to voicepack (.pt)",
    )
    parser.add_argument(
        "--output-dir",
        default="test_output/",
        help="Directory to save generated WAV files",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on (default: auto)",
    )

    args = parser.parse_args()

    model_path = args.model

    run_inference(
        model_path=model_path,
        voicepack_path=args.voicepack,
        output_dir=args.output_dir,
        device=args.device,
    )

if __name__ == "__main__":
    main()
