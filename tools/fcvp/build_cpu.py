#!/usr/bin/env python3
from pathlib import Path
import argparse
from reference import build_bridge

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the CPU oracle and production MM solver against pinned NEP_CPU"
    )
    parser.add_argument("--nep-cpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("local/fcvp/libfcvp.so"))
    args = parser.parse_args()
    build_bridge(args.nep_cpu.resolve(), args.output.resolve())
    print(args.output)
