#!/usr/bin/env python3
"""Fetch checksum-pinned, publicly available PbTe XYZ examples (no authentication)."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

COMMIT = "068db4d649d6c71d56981aae7977b33c402c707d"
BASE = f"https://raw.githubusercontent.com/brucefan1983/GPUMD-Tutorials/{COMMIT}/examples/11_NEP_potential_PbTe/"
FILES = {
    "train.xyz": "3b9f1d55fe61def164ab87370dce66b760d6740d3c1cc35ec6bc3db057572c5c",
    "test.xyz": "ca3cf9cb4bff11d6f37ce466db3c52352c0d682f6370b73b8a9ea963f7f100f4",
}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    for name, expected in FILES.items():
        with urllib.request.urlopen(BASE + name, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("Checksum mismatch for " + name)
        (a.output / name).write_bytes(data)
        print(name, expected)
