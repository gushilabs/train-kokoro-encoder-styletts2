#!/usr/bin/env python3
"""
Kokoro: Test Inference
==============================
Tests the fine-tuned Kokoro model with a  phonetic test set.

Pre-requisites:
    pip install -q kokoro>=0.9.4 soundfile
    apt-get -qq -y install espeak-ng > /dev/null 2>&1

Usage:
    python test_inference.py \
        --checkpoint ~/Downloads/models/Kokoro2nd-LibriTTS/epoch_2nd_00000.pth \
        --voicepack voices/am_2nd_speaker78.pt \
        --output-dir test_output/epoch0

"""

import argparse
import sys
from pathlib import Path

# Prefer the kokoro submodule over any pip-installed kokoro package
#_repo_root = Path(__file__).resolve().parents[1]
#_kokoro_submodule = _repo_root / "kokoro"
#if _kokoro_submodule.exists() and str(_kokoro_submodule) not in sys.path:
#    sys.path.insert(0, str(_kokoro_submodule))

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Printing, in the only sense with which we are at present concerned, differs from most arts.",
    "Why did you do that? That is absolutely incredible!",
    "The total cost is exactly one hundred twenty-three million dollars.",
    "She sells seashells by the seashore.",
    "Are you serious right now? I can't believe it!",
    "A swift red fox gracefully cleared the sleeping dog."
]

def convert_checkpoint(checkpoint_path: str, output_path: str) -> str:
    """Convert a StyleTTS2 Stage 2 checkpoint to Kokoro KModel format.

    Extracts the 5 inference components (bert, bert_encoder, predictor,
    text_encoder, decoder) from the training checkpoint. All state dict
    keys must have the 'module.' prefix for KModel's loading fallback
    to work correctly.

    Requires that training was done with the new parametrizations API
    (torch.nn.utils.parametrizations.weight_norm/spectral_norm) so the
    state dict keys are natively compatible with Kokoro's KModel.
    """
    import torch

    print(f"Converting checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net = ckpt["net"]

    def ensure_module_prefix(state_dict):
        """Ensure all keys have 'module.' prefix for KModel compatibility."""
        return {
            ("module." + k if not k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    kokoro_weights = {}
    for key in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]:
        if key in net:
            kokoro_weights[key] = ensure_module_prefix(net[key])
            print(f"  {key}: {len(kokoro_weights[key])} keys")
        else:
            print(f"  WARNING: '{key}' not found in checkpoint")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kokoro_weights, str(output))
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  Saved Kokoro-format weights: {output} ({size_mb:.1f} MB)")
    return str(output)


def run_inference(
    model_path: str,
    voicepack_path: str,
    config_path: str,
    output_dir: str,
    device: str = "auto",
):
    """Run inference on the German test set."""
    import torch
    import soundfile as sf
    from kokoro import KModel, KPipeline

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model with our fine-tuned weights and config
    print(f"Loading model from: {model_path}")
    #print(f"  Config: {config_path}")
    #kmodel = KModel(repo_id="hexgrad/Kokoro-82M", config=config_path, model=model_path)
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", model=model_path)
    kmodel = kmodel.to(device).eval()

    # Create pipeline with German lang_code
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
        "--checkpoint",
        help="Path to StyleTTS2 checkpoint (.pth) — will be converted automatically",
    )
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
        "--config",
        default="training/config.json",
        help="Path to Kokoro config.json",
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

    # Convert checkpoint if needed
    if args.checkpoint:
        model_path = convert_checkpoint(
            args.checkpoint,
            str(Path(args.output_dir) / "kokoro_trained_converted.pth"),
        )
    else:
        model_path = args.model

    run_inference(
        model_path=model_path,
        voicepack_path=args.voicepack,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Clean up temporary model file
    if args.checkpoint:
        converted_file = Path(model_path)
        if converted_file.exists():
            converted_file.unlink()
            print(f"Cleaned up temporary model file: {converted_file}")

if __name__ == "__main__":
    main()
