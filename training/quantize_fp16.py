#!/usr/bin/env python
"""Convert shipped model safetensors to fp16 in place (halves size).

Inference upcasts fp16 -> fp32 on load (``load_state_dict`` copies with a dtype
cast), so this is lossless for our greedy/beam argmax decoding. Idempotent:
already-fp16 tensors are left as-is.

    python training/quantize_fp16.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIRS = [
    REPO_ROOT / "indicate" / "data" / "hindi_to_english" / "saved_weights",
    REPO_ROOT / "indicate" / "data" / "punjabi_to_english" / "saved_weights",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs", nargs="*", type=Path, default=DEFAULT_DIRS)
    args = parser.parse_args()
    for d in args.dirs:
        for f in sorted(d.glob("*.safetensors")):
            sd = load_file(str(f))
            before = f.stat().st_size
            sd = {
                k: (v.half() if v.dtype == torch.float32 else v) for k, v in sd.items()
            }
            save_file(sd, str(f))
            after = f.stat().st_size
            print(f"  {f.relative_to(REPO_ROOT)}: {before / 1e6:.1f}MB -> {after / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
