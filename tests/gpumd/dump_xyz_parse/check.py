#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
GPUMD = os.path.abspath(os.path.join(ROOT, "../../../src/gpumd"))
MODEL = os.path.abspath(os.path.join(ROOT, "../dump_xyz_stress/isolated/model.xyz"))
POT = "../../../../potentials/lj/Ar_10A.txt"


def write_case(name, dump_line):
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    shutil.copy(MODEL, os.path.join(d, "model.xyz"))
    with open(os.path.join(d, "run.in"), "w") as f:
        f.write("potential       %s\n" % POT)
        f.write("time_step       0.0\n")
        f.write("ensemble        nve\n")
        f.write("%s\n" % dump_line)
        f.write("run             1\n")
    return d


def run_gpumd(d):
    log = os.path.join(d, "gpumd.log")
    with open(log, "w") as out:
        proc = subprocess.run([GPUMD], cwd=d, stdout=out, stderr=subprocess.STDOUT)
    with open(log) as f:
        text = f.read()
    return proc.returncode, text


def expect_fail(name, dump_line, needle):
    d = write_case(name, dump_line)
    code, text = run_gpumd(d)
    if code == 0:
        sys.exit("%s: expected failure, got exit 0" % name)
    if needle not in text:
        sys.exit("%s: missing %r in log" % (name, needle))
    print("%s: PASS fail-as-expected" % name)


def expect_ok(name, dump_line):
    d = write_case(name, dump_line)
    code, text = run_gpumd(d)
    if "Finished running GPUMD" not in text and code != 0:
        sys.exit("%s: unexpected failure\n%s" % (name, text[-500:]))
    if "Unknown identifier" in text or "should be followed" in text:
        sys.exit("%s: parse error on valid input" % name)
    print("%s: PASS" % name)


def main():
    expect_fail(
        "unknown",
        "dump_xyz        -1 0 1 out.xyz foo",
        "Unknown identifier",
    )
    expect_fail(
        "partial_rm",
        "dump_xyz        -1 0 1 out.xyz volume 3.5 128 pressure 4.0 bad",
        "number of Voronoi directions",
    )
    expect_fail(
        "missing_m",
        "dump_xyz        -1 0 1 out.xyz pressure 4.0",
        "followed by the number of directions",
    )
    expect_fail(
        "conflict_rm",
        "dump_xyz        -1 0 1 out.xyz volume 3.5 128 pressure 4.0 128",
        "must use the same R",
    )
    expect_ok(
        "ok_shared",
        "dump_xyz        -1 0 1 out.xyz volume 3.0 128 pressure",
    )
    expect_ok(
        "ok_own",
        "dump_xyz        -1 0 1 out.xyz pressure 3.0 128",
    )


if __name__ == "__main__":
    main()
