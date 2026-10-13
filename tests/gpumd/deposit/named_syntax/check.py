#!/usr/bin/env python3
"""Black-box regression suite for named, staged deposition commands.

Every case is written to a fresh output directory. No tracked fixture, potential,
or source file is modified. This script needs only Python's standard library.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


BASE = Path(__file__).resolve().parents[4]
PLANE = "region plane normal 0 0 1 bounds 0 10 0 10 height 20"
SURFACE_ALL = "region surface normal 0 0 1 bounds 0 10 0 10 height local radius 5 gap 2 scope all"
SURFACE_MOBILE = "region surface normal 0 0 1 bounds 0 10 0 10 height local radius 5 gap 2 scope mobile"
VECTOR = "velocity vector 0 0 -0.1 units angstrom/fs"
BEAM = "velocity beam speed 0.1 units angstrom/fs direction 0 0 -1"
ATOMS = [("C", 2, 2, 0, 0), ("C", 8, 2, 0, 0),
         ("C", 2, 8, 0, 0), ("C", 8, 8, 0, 0)]
SCOPE_ATOMS = [("C", 2, 2, 2, 0), ("C", 8, 2, 2, 0),
               ("C", 2, 8, 2, 0), ("C", 8, 8, 2, 0),
               ("C", 5, 5, 12, 1)]


def model(atoms=ATOMS, box=(20, 20, 60), pbc="T T F"):
    lattice = (f"{box[0]} 0 0 0 {box[1]} 0 0 0 {box[2]}" if len(box) == 3
               else " ".join(str(value) for value in box))
    lines = [str(len(atoms)),
             f'pbc="{pbc}" Lattice="{lattice}" '
             'Properties=species:S:1:pos:R:3:vel:R:3:group:I:1']
    lines += [f"{species} {x} {y} {z} 0 0 0 {group}"
              for species, x, y, z, group in atoms]
    return "\n".join(lines) + "\n"


def stage(name="d", particle="particle atom C", placement="placement point at 5 5",
          region=PLANE, velocity=VECTOR, extras=()):
    return [f"deposit {name} {part}" for part in
            (particle, placement, region, velocity, *extras)]


def emit(name="d", count="1"):
    return f"deposit {name} emit number {count}"


def pulse(name="d", count=1, wait="entry", relax=0, max_wait=200):
    return (f"deposit {name} pulse number {count} wait {wait} "
            f"relax {relax} max_wait {max_wait}")


def xyz(path):
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    if len(lines) < 2:
        raise ValueError(f"XYZ lacks a header: {path}")
    count = int(lines[0])
    header = lines[1]
    props = re.search(r"\bProperties=(\S+)", header)
    if not props:
        raise ValueError("XYZ header lacks Properties")
    fields = props.group(1).split(":")
    offsets = {}
    offset = 0
    for i in range(0, len(fields), 3):
        offsets[fields[i]] = (offset, int(fields[i + 2]))
        offset += int(fields[i + 2])
    atoms = []
    for line in lines[2:2 + count]:
        parts = line.split()
        atom = {}
        for key, (start, width) in offsets.items():
            values = parts[start:start + width]
            atom[key] = values if key == "species" else [float(v) for v in values]
        atoms.append(atom)
    if len(atoms) != count:
        raise ValueError(f"XYZ declared {count} atoms, found {len(atoms)}")
    tm = re.search(r"\bTime=([-+\deE.]+)", header)
    return {"atoms": atoms, "time": float(tm.group(1)) if tm else None}


def close(a, b, tol=2e-5):
    return abs(a - b) <= tol


def point(snapshot, index):
    return snapshot["atoms"][index]["pos"]


def velocity(snapshot, index):
    return snapshot["atoms"][index]["vel"]


def check_case(case, code, log, before, after):
    errors = []
    details = {"exit_code": code}
    if case.get("reject"):
        if code in (None, 0):
            errors.append("expected command rejection")
        if "Input Error" not in log and "ERROR" not in log and "Error" not in log:
            errors.append("failure lacked an input/runtime error diagnostic")
        pattern = case.get("error")
        if pattern and not re.search(pattern, log, re.I):
            errors.append(f"missing diagnostic /{pattern}/")
        return errors, details
    if code != 0 or "Finished running GPUMD" not in log:
        errors.append("engine did not finish successfully")
        return errors, details
    if after is None:
        errors.append("missing after.xyz")
        return errors, details
    atoms = case.get("atoms", ATOMS)
    expected = len(atoms) + case.get("added", 0)
    details["atom_count"] = len(after["atoms"])
    if len(after["atoms"]) != expected:
        errors.append(f"expected {expected} atoms, got {len(after['atoms'])}")
        return errors, details
    if "new_pos" in case:
        actual = point(after, len(atoms))
        details["new_pos"] = actual
        if any(not close(a, b, case.get("pos_tol", 2e-4))
               for a, b in zip(actual, case["new_pos"])):
            errors.append(f"position {actual} != {case['new_pos']}")
    if "new_positions" in case:
        actual = [atom["pos"] for atom in after["atoms"][len(atoms):]]
        details["new_positions"] = actual
        if len(actual) != len(case["new_positions"]) or any(
                any(not close(x, y, case.get("pos_tol", 2e-4)) for x, y in zip(a, b))
                for a, b in zip(actual, case["new_positions"])):
            errors.append(f"positions {actual} != {case['new_positions']}")
    if "new_velocity" in case:
        actual = velocity(after, len(atoms))
        details["new_velocity"] = actual
        if any(not close(a, b, case.get("vel_tol", 2e-5))
               for a, b in zip(actual, case["new_velocity"])):
            errors.append(f"velocity {actual} != {case['new_velocity']}")
    if "new_species" in case:
        actual = after["atoms"][len(atoms)]["species"][0]
        details["new_species"] = actual
        if actual != case["new_species"]:
            errors.append(f"species {actual} != {case['new_species']}")
    if case.get("unchanged"):
        if before is None or len(before["atoms"]) != len(after["atoms"]):
            errors.append("config-only run changed atom count")
        elif any(any(not close(x, y) for x, y in
                         zip(a["pos"], b["pos"]))
                 for a, b in zip(before["atoms"], after["atoms"])):
            errors.append("config-only run changed positions")
    if "time" in case and not close(after["time"], case["time"], 2e-4):
        errors.append(f"time {after['time']} != {case['time']} fs")
    if "log" in case and not re.search(case["log"], log, re.I):
        errors.append(f"missing log /{case['log']}/")
    match = re.search(r"Deposit pulse completed:\s*wait_steps=(\d+)\s+relax_steps=(\d+)\s+md_steps=(\d+)\s+md_time_fs=([-+\deE.]+)", log)
    if "pulse_steps" in case:
        if not match:
            errors.append("missing pulse completion accounting")
        else:
            observed = tuple(int(match.group(i)) for i in range(1, 4))
            details["pulse_steps"] = observed
            if observed != tuple(case["pulse_steps"]):
                errors.append(f"pulse steps {observed} != {case['pulse_steps']}")
    if case.get("pulse_early"):
        if not match or int(match.group(1)) >= case["max_wait"]:
            errors.append("pulse did not report entry before max_wait")
        elif int(match.group(2)) != case["relax"]:
            errors.append("pulse relaxation step count differs from request")
        elif int(match.group(3)) != int(match.group(1)) + int(match.group(2)):
            errors.append("pulse MD step accounting is inconsistent")
        elif not close(float(match.group(4)), int(match.group(3)) * case["step"]):
            errors.append("pulse MD time accounting is inconsistent")
    if case.get("no_pulse_dump"):
        dump = case["directory"] / "zero_pulse.xyz"
        if dump.exists() and dump.read_text(encoding="utf-8").strip():
            errors.append("zero-step pulse produced a step-based dump frame")
    return errors, details


def cases():
    c = []
    def add(name, commands, **kwargs):
        c.append(dict(name=name, commands=commands, **kwargs))

    add("config_only", stage() + ["dump_xyz -1 0 1 after.xyz velocity", "run 1"], unchanged=True)
    add("point_plane_vector", stage() + [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        added=1, new_pos=[5, 5, 20], new_velocity=[0, 0, -0.1])
    add("flight_off", stage(velocity="velocity vector 0 0 -1 units angstrom/fs",
                            extras=("flight skip off",)) +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        added=1, step=1, new_pos=[5, 5, 19], new_velocity=[0, 0, -1], time=1)
    add("flight_default", stage(velocity="velocity vector 0 0 -1 units angstrom/fs") +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        added=1, step=1, new_pos=[5, 5, 19], new_velocity=[0, 0, -1], time=1)
    add("flight_on", stage(velocity="velocity vector 0 0 -1 units angstrom/fs",
                           extras=("flight skip on clearance auto",)) +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        added=1, step=1, new_velocity=[0, 0, -1])
    grid_plane = "region plane normal 0 0 1 bounds 0 5 0 5 height 20"
    add("grid_sequential_all", stage(placement="placement grid spacing 5 5 select sequential",
                                     region=grid_plane) +
        [emit(count="all"), "dump_xyz -1 0 1 after.xyz velocity", "run 1"], added=4)
    add("grid_random", stage(placement="placement grid spacing 5 5 select random",
                             region=grid_plane, extras=("seed 19",)) +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"], added=1)
    add("plane_basis", stage(region=PLANE + " basis 1 0 0") +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        added=1, new_pos=[5, 5, 20])
    add("random_plane", stage(placement="placement random attempt 100", extras=("seed 19",)) +
        [emit(count="2"), "dump_xyz -1 0 1 after.xyz velocity", "run 1"], added=2)
    add("gaussian_beam", stage(placement="placement gaussian center 5 5 sigma 0.2 attempt 100",
                               velocity=BEAM, extras=("seed 19",)) +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"], added=1,
        new_velocity=[0, 0, -0.1])
    add("box_random", stage(placement="placement random attempt 100",
                            region="region box bounds 3 7 3 7 20 22", extras=("seed 19",)) +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"], added=1)
    add("mixture", stage(particle="particle mixture C 0.5 Si 0.5", extras=("seed 19",)) +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        potential="potentials/tersoff/SiC_tersoff_1989.txt", added=1)
    add("mixture_first_only", stage(particle="particle mixture C 1 Si 0", extras=("seed 19",)) +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        potential="potentials/tersoff/SiC_tersoff_1989.txt", added=1, new_species="C")
    add("mixture_last_only", stage(particle="particle mixture C 0 Si 1", extras=("seed 19",)) +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        potential="potentials/tersoff/SiC_tersoff_1989.txt", added=1, new_species="Si")
    add("molecule", stage(particle="particle molecule C.xyz") +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"], added=1)
    add("molecule_two_atoms", stage(particle="particle molecule C2.xyz") +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        added=2, new_positions=[[4.25, 5, 20], [5.75, 5, 20]])
    add("thermal", stage(velocity="velocity thermal temperature 0 drift 0 0 -0.1 units angstrom/fs") +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        added=1, new_velocity=[0, 0, -0.1])
    add("named_isolation", stage("a", placement="placement point at 4 4") +
        stage("b", placement="placement point at 6 6") +
        [emit("a"), emit("b"), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        added=2, new_positions=[[4, 4, 20], [6, 6, 20]])
    random_stage = stage(placement="placement random attempt 100", extras=("seed 77",))
    add("seed_batch", random_stage +
        [emit(count="2"), "dump_xyz -1 0 1 after.xyz velocity", "run 1"], added=2)
    add("seed_split", random_stage +
        [emit(), emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"], added=2)
    add("seed_repeat", random_stage +
        [emit(count="2"), "dump_xyz -1 0 1 after.xyz velocity", "run 1"], added=2)
    add("facet_replacement", stage(placement="placement point at 4 4") +
        ["deposit d placement point at 6 6", emit(),
         "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        added=1, new_pos=[6, 6, 20])
    add("surface_all", ["fix 1"] + stage(region=SURFACE_ALL) +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        atoms=SCOPE_ATOMS, added=1, new_pos=[5, 5, 14])
    add("surface_global", ["fix 1"] +
        stage(region="region surface normal 0 0 1 bounds 0 10 0 10 height global gap 2 scope all") +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        atoms=SCOPE_ATOMS, added=1, new_pos=[5, 5, 14])
    add("surface_mobile", ["fix 1"] + stage(region=SURFACE_MOBILE) +
        [emit(), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        atoms=SCOPE_ATOMS, added=1, new_pos=[5, 5, 4])
    add("surface_scope_sequence", ["fix 1"] + stage("a", region=SURFACE_ALL) +
        stage("b", region=SURFACE_MOBILE) +
        [emit("a"), emit("b"), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        atoms=SCOPE_ATOMS, added=2, new_positions=[[5, 5, 14], [5, 5, 16]])
    add("surface_scope_reverse", ["fix 1"] + stage("a", region=SURFACE_MOBILE) +
        stage("b", region=SURFACE_ALL) +
        [emit("a"), emit("b"), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        atoms=SCOPE_ATOMS, added=2, new_positions=[[5, 5, 4], [5, 5, 14]])

    bad = [
        ("missing_particle", stage()[1:], "particle"),
        ("missing_placement", [stage()[0]] + stage()[2:], "placement"),
        ("missing_region", stage()[:2] + stage()[3:], "region"),
        ("missing_velocity", stage()[:3], "velocity"),
        ("bad_units", stage(velocity="velocity vector 0 0 -1 units nm/ps"), "units"),
        ("nan_velocity", stage(velocity="velocity vector 0 0 nan units angstrom/fs"), "velocity|finite|number"),
        ("nan_height", stage(region=PLANE.replace("height 20", "height nan")), "height|finite|number"),
        ("box_point", stage(region="region box bounds 3 7 3 7 20 22"), "box|random"),
        ("legacy", ["deposit point atom C origin 5 5 direction axis -z surface fixed 20 velocity constant 0.1"], "deposit"),
        ("unknown_name", [emit("missing")], "missing|unknown|undefined"),
        ("zero_count", stage() + [emit(count="0")], "number|count"),
        ("bad_mixture", stage(particle="particle mixture C 0.2 C 0.2"), "mixture|fraction|sum"),
        ("skip_zero_velocity", stage(velocity="velocity vector 0 0 0 units angstrom/fs",
                                      extras=("flight skip on clearance auto",)), "flight|velocity|speed"),
        ("skip_parallel_velocity", stage(velocity="velocity vector 1 0 0 units angstrom/fs",
                                          extras=("flight skip on clearance auto",)), "flight|direction|velocity"),
    ]
    for name, commands, diagnostic in bad:
        add(name, commands + [emit()] if name not in ("legacy", "unknown_name", "zero_count") else commands,
            reject=True, error=diagnostic)
    add("pulse_early_entry", stage(velocity="velocity vector 0 0 -4 units angstrom/fs") +
        [pulse(relax=2, max_wait=50), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        added=1, step=1, pulse_early=True, relax=2, max_wait=50)
    add("pulse_skip_on", stage(velocity="velocity vector 0 0 -1 units angstrom/fs",
                               extras=("flight skip on clearance auto",)) +
        [pulse(relax=2, max_wait=50), "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        added=1, step=1, pulse_steps=[1, 2, 3], time=4)
    add("pulse_all_entry", stage(placement="placement grid spacing 5 5 select sequential",
                                 region="region plane normal 0 0 1 bounds 0 5 0 5 height 20",
                                 velocity="velocity vector 0 0 -4 units angstrom/fs") +
        [pulse(count="all", relax=0, max_wait=50),
         "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        added=4, step=1, pulse_early=True, relax=0, max_wait=50)
    add("pulse_molecule_all_atoms", stage(particle="particle molecule C2z.xyz",
                                          velocity="velocity vector 0 0 -4 units angstrom/fs") +
        ["dump_xyz -1 0 1 after.xyz velocity", pulse(relax=0, max_wait=10)],
        added=2, step=1, pulse_steps=[4, 0, 4], pulse_early=True,
        relax=0, max_wait=10)
    add("pulse_timeout", stage(velocity="velocity vector 0 0 0.1 units angstrom/fs") +
        [pulse(max_wait=3)], reject=True, error="max_wait|entry")
    add("pulse_zero_step", stage(region=PLANE.replace("height 20", "height 5")) +
        ["dump_xyz -1 0 1 zero_pulse.xyz velocity", pulse(relax=0, max_wait=3),
         "dump_xyz -1 0 1 after.xyz velocity", "run 1"],
        added=1, step=1, new_pos=[5, 5, 4.9], time=1,
        pulse_steps=[0, 0, 0], no_pulse_dump=True)
    add("pulse_without_ensemble", stage() + [pulse()],
        omit_ensemble=True, step=1, reject=True, error="ensemble|pulse")
    add("skip_external_force", stage(extras=("flight skip on clearance auto",)) +
        ["add_force 0 0 0 0 0.1", emit()],
        step=1, reject=True, error="external forces|fields|flight")
    add("pulse_nvt_ramp", ["ensemble nvt_ber 300 400 100"] + stage() + [pulse()],
        step=1, reject=True, error="temperature ramp|pulse")
    add("pulse_npt", ["ensemble npt_ber 300 300 100 0 0 0 53 53 53 2000"] +
        stage() + [pulse()], step=1, reject=True, error="NVE|NVT|ensemble|pulse")
    add("skip_tilted_normal", stage(region=PLANE.replace("normal 0 0 1", "normal 1 0 1"),
                                     extras=("flight skip on clearance auto",)) +
        [emit()], step=1, reject=True, error="Cartesian normal|flight")
    add("skip_periodic_normal", stage(extras=("flight skip on clearance auto",)) +
        [emit()], pbc="T T T", step=1, reject=True, error="nonperiodic|flight")
    add("skip_tilted_box", stage(extras=("flight skip on clearance auto",)) +
        [emit()], box=(20, 0, 0, 1, 20, 0, 0, 0, 60),
        step=1, reject=True, error="orthogonal|flight")
    add("skip_unsupported_potential", stage(extras=("flight skip on clearance auto",)) +
        [emit()], potential="potentials/tersoff/C_Tersoff_1989.txt",
        step=1, reject=True, error="finite.range|unsupported|clearance|does not support")
    return c


def run_case(case, directory, engine, root, potential, gpu, timeout):
    directory.mkdir()
    case["directory"] = directory
    (directory / "model.xyz").write_text(model(case.get("atoms", ATOMS),
                                                case.get("box", (20, 20, 60)),
                                                case.get("pbc", "T T F")), encoding="utf-8")
    source_potential = (root / case["potential"] if "potential" in case
                        else potential)
    shutil.copyfile(source_potential, directory / "potential.txt")
    (directory / "C.xyz").write_text("1\nCarbon molecule\nC 0 0 0\n", encoding="utf-8")
    (directory / "C2.xyz").write_text("2\nCarbon dimer\nC -0.75 0 0\nC 0.75 0 0\n",
                                      encoding="utf-8")
    (directory / "C2z.xyz").write_text("2\nVertical carbon dimer\nC 0 0 -0.75\nC 0 0 0.75\n",
                                       encoding="utf-8")
    needs_time = any(" flight skip on" in command or " pulse " in command
                     for command in case["commands"])
    step = case.get("step", 1 if needs_time else 0)
    before_path = directory / "before.xyz"
    lines = ["potential ./potential.txt", f"time_step {step}"]
    if not case.get("omit_ensemble"):
        lines.append("ensemble nve")
    if case.get("unchanged"):
        lines += ["dump_xyz -1 0 1 before.xyz velocity", "run 1"]
    lines += case["commands"]
    if case.get("reject"):
        lines += ["dump_xyz -1 0 1 after.xyz velocity", "run 1"]
    (directory / "run.in").write_text("\n".join(lines) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        completed = subprocess.run([str(engine)], cwd=directory, env=env,
                                   capture_output=True, text=True, timeout=timeout)
        code, log = completed.returncode, completed.stdout + "\n" + completed.stderr
    except subprocess.TimeoutExpired as exc:
        code = None
        out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        log = out + "\n" + err + "\nTIMEOUT"
    (directory / "gpumd.log").write_text(log, encoding="utf-8")
    before = xyz(before_path)
    after = xyz(directory / "after.xyz")
    errors, details = check_case(case, code, log, before, after)
    return {"name": case["name"], "passed": not errors,
            "errors": errors, **details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=BASE,
                        help="repository root used to resolve the default carbon NEP")
    parser.add_argument("--repo", type=Path,
                        help="alias for --root; useful from a separate build checkout")
    parser.add_argument("--potential", type=Path,
                        help="override default potentials/nep/C_2022_NEP3.txt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--case", action="append", help="run only a named case; repeatable")
    args = parser.parse_args()
    root = (args.repo or args.root).resolve(strict=True)
    engine = args.engine.resolve(strict=True)
    potential = (args.potential or root / "potentials/nep/C_2022_NEP3.txt").resolve(strict=True)
    output = args.output.resolve()
    if args.gpu < 0 or args.timeout < 1:
        parser.error("--gpu must be nonnegative and --timeout positive")
    if output.exists():
        parser.error("--output must be a new directory")
    if root == output or root in output.parents:
        parser.error("--output must be outside the repository")
    if not engine.is_file() or not potential.is_file():
        parser.error("engine and potential must be files")
    selected = [case for case in cases() if not args.case or case["name"] in args.case]
    if args.case and len(selected) != len(set(args.case)):
        parser.error("unknown or duplicate --case name")
    output.mkdir(parents=True)
    report = {"engine": str(engine), "repo": str(root), "potential": str(potential),
              "gpu": args.gpu, "cases": []}
    for case in selected:
        try:
            result = run_case(case, output / case["name"], engine, root, potential,
                              args.gpu, args.timeout)
        except Exception as exc:
            result = {"name": case["name"], "passed": False,
                      "errors": [f"harness failure: {type(exc).__name__}: {exc}"]}
        report["cases"].append(result)
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{case['name']}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    if not args.case:
        off_bytes = (output / "flight_off" / "after.xyz").read_bytes() if (
            output / "flight_off" / "after.xyz").exists() else None
        default_bytes = (output / "flight_default" / "after.xyz").read_bytes() if (
            output / "flight_default" / "after.xyz").exists() else None
        equal = off_bytes is not None and off_bytes == default_bytes
        result = {"name": "flight_default_equals_off", "passed": equal,
                  "errors": [] if equal else ["default and explicit-off XYZ differ"]}
        report["cases"].append(result)
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"flight_default_equals_off: {'PASS' if equal else 'FAIL'}", flush=True)
        seed_paths = [output / name / "after.xyz" for name in
                      ("seed_batch", "seed_split", "seed_repeat")]
        snapshots = [xyz(path) for path in seed_paths]
        if all(snapshots):
            positions = [[atom["pos"] for atom in snap["atoms"][-2:]]
                         for snap in snapshots]
            errors = []
            for name, candidate in zip(("seed_split", "seed_repeat"), positions[1:]):
                if any(any(not close(x, y) for x, y in zip(a, b))
                       for a, b in zip(positions[0], candidate)):
                    errors.append(f"{name} differs from seed_batch")
            result = {"name": "seed_progression", "passed": not errors,
                      "errors": errors, "positions": positions}
        else:
            result = {"name": "seed_progression", "passed": False,
                      "errors": ["missing seed XYZ"]}
        report["cases"].append(result)
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"seed_progression: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
        off = xyz(output / "flight_off" / "after.xyz")
        on = xyz(output / "flight_on" / "after.xyz")
        if off and on:
            a, b = off["atoms"][-1], on["atoms"][-1]
            shift = a["pos"][2] - b["pos"][2]
            expected_shift = 20 - 2 * 4.2
            errors = []
            if not close(shift, expected_shift, 2e-3):
                errors.append(f"flight shift {shift} != {expected_shift} A")
            if any(not close(x, y) for x, y in zip(a["vel"], b["vel"])):
                errors.append("flight skip changed launch velocity")
            if not close(off["time"], on["time"]):
                errors.append("flight skip changed the MD clock")
            result = {"name": "flight_pair", "passed": not errors,
                      "errors": errors, "shift_angstrom": shift,
                      "off_time_fs": off["time"], "on_time_fs": on["time"]}
        else:
            result = {"name": "flight_pair", "passed": False,
                      "errors": ["missing flight off/on XYZ"]}
        report["cases"].append(result)
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"flight_pair: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    return 0 if all(result["passed"] for result in report["cases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
