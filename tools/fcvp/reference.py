"""FCVP CPU oracle and independent dense MM reference (NumPy/SciPy).

Energy and virial are per atom; forces use one row per Cartesian component.
This module fits a frozen representation. It does not claim a GPU training speedup.
"""

from __future__ import annotations
import ctypes as C
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import numpy as np
from scipy.optimize import minimize

NEP_CPU_COMMIT = "af615ce0e81bbb860e77923762c505567affa8d1"
DP = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
IP = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")


def build_bridge(nep_cpu: Path, output: Path):
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(nep_cpu), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != NEP_CPU_COMMIT:
        raise ValueError(f"Expected pinned NEP_CPU {NEP_CPU_COMMIT}, got {revision}")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O3",
            "-shared",
            "-fPIC",
            "-I" + str(nep_cpu / "src"),
            "-I" + str(root / "src"),
            str(Path(__file__).with_name("nep_cpu_bridge.cpp")),
            *map(
                str,
                [
                    nep_cpu / "src" / f
                    for f in ("nep.cpp", "ewald_nep.cpp", "neighbor_nep.cpp")
                ],
            ),
            "-o",
            str(output),
        ],
        check=True,
    )
    metadata = {
        "nep_cpu_commit": revision,
        "binary_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(
            ["g++", "--version"], text=True
        ).splitlines()[0],
        "flags": "-std=c++17 -O3 -shared -fPIC",
        "solver_sha256": hashlib.sha256(
            (root / "src/main_gnep/fcvp_solver.cuh").read_bytes()
        ).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


@dataclass
class Model:
    header: list[str]
    values: np.ndarray
    indices: np.ndarray
    parameter_count: int
    elements: list[str]
    shared_bias_index: int | None = None

    @classmethod
    def read(cls, path):
        lines = Path(path).read_text().splitlines()
        first = lines[0].split()
        if first[0] not in ("nep3", "nep4", "nep5", "nep3_zbl", "nep4_zbl", "nep5_zbl"):
            raise ValueError(
                "Only ordinary one-hidden-layer NEP3/4/5 models are supported"
            )
        version, types = int(first[0][3]), int(first[1])
        if types != len(first) - 2 or types < 1:
            raise ValueError("Invalid element list")
        end = next(i for i, l in enumerate(lines) if l.split()[0] == "ANN") + 1
        fields = {l.split()[0]: l.split()[1:] for l in lines[1:end]}
        neuron, second = map(int, fields["ANN"])
        if second != 0 or neuron < 1:
            raise ValueError("Only one hidden layer is supported")
        nr, na = map(int, fields["n_max"])
        ls = list(map(int, fields["l_max"]))
        dim = nr + 1 + (na + 1) * (ls[0] + sum(v > 0 for v in ls[1:]))
        block = (dim + 2) * neuron + (version == 5)
        networks = 1 if version == 3 else types
        indices = [
            t * block + (dim + 1) * neuron + j
            for t in range(networks)
            for j in range(neuron + (version == 5))
        ]
        ann = block * networks + 1
        # NEP5's additional shared bias is a gauge redundancy. Keep it fixed;
        # the per-type biases already span the same constant column.
        if version != 5:
            indices.append(ann - 1)
        br, ba = map(int, fields["basis_size"])
        parameters = ann + types * types * ((nr + 1) * (br + 1) + (na + 1) * (ba + 1))
        values = np.array([float(l.strip()) for l in lines[end:]], dtype=np.float64)
        extra = 0
        if "_zbl" in first[0] and list(map(float, fields["zbl"][:2])) == [0, 0]:
            extra = 10 * types * (types + 1) // 2
        if len(values) != parameters + dim + extra or not np.isfinite(values).all():
            raise ValueError("Unexpected model size or nonfinite parameters")
        return cls(
            lines[:end],
            values,
            np.array(indices, np.int32),
            parameters,
            first[2:],
            ann - 1 if version == 5 else None,
        )

    @property
    def beta(self):
        return self.values[self.indices].copy()

    @property
    def training_parameter_count(self):
        # GNEP does not train NEP5's additional serialized shared bias.
        return self.parameter_count - (self.shared_bias_index is not None)

    def fixed_regularization_parameters(self):
        fixed = self.values[: self.parameter_count].copy()
        fixed[self.indices] = 0
        if self.shared_bias_index is not None:
            fixed = np.delete(fixed, self.shared_bias_index)
        return fixed

    def write(self, path, beta):
        values = self.values.copy()
        values[self.indices] = beta
        Path(path).write_text(
            "\n".join(self.header)
            + "\n"
            + "\n".join(f"{x:.17e}" for x in values)
            + "\n"
        )


@dataclass
class Frame:
    species: list[str]
    positions: np.ndarray
    cell: np.ndarray
    energy: float
    force: np.ndarray
    virial: np.ndarray | None
    weight: float
    text: str


def read_xyz(path):
    frames = []
    lines = Path(path).read_text().splitlines(keepends=True)
    start = 0
    while start < len(lines):
        if not lines[start].strip():
            start += 1
            continue
        n = int(lines[start])
        stop = start + n + 2
        if n < 1 or stop > len(lines):
            raise ValueError("Truncated or empty XYZ frame")
        fields = {
            k.lower(): v.strip("\"'")
            for k, v in re.findall(
                r'(\w+)\s*=\s*("[^"]*"|\'[^\']*\'|[^\s]+)', lines[start + 1]
            )
        }
        pbc = fields.get("pbc", "T T T").upper().split()
        if pbc not in (["T", "T", "T"], ["1", "1", "1"], ["TRUE", "TRUE", "TRUE"]):
            raise ValueError(
                "The NEP_CPU oracle currently requires fully periodic frames"
            )
        cell = np.array(list(map(float, fields["lattice"].split()))).reshape(3, 3)
        if abs(np.linalg.det(cell)) < 1e-12:
            raise ValueError("Singular cell")
        props = fields["properties"].split(":")
        offset = 0
        columns = {}
        if len(props) % 3:
            raise ValueError("Invalid Properties")
        for i in range(0, len(props), 3):
            count = int(props[i + 2])
            columns[props[i].lower()] = slice(offset, offset + count)
            offset += count
        rows = [line.split() for line in lines[start + 2 : stop]]
        if any(len(row) != offset for row in rows):
            raise ValueError("Invalid atom row width")
        species = [row[columns["species"]][0] for row in rows]
        positions = np.array([row[columns["pos"]] for row in rows], float)
        key = "force" if "force" in columns else "forces"
        forces = np.array([row[columns[key]] for row in rows], float)
        if positions.shape != (n, 3) or forces.shape != (n, 3):
            raise ValueError("Expected three components")
        virial = None
        if "virial" in fields:
            virial = np.array(list(map(float, fields["virial"].split()))).reshape(3, 3)
        if "stress" in fields:
            stress = -abs(np.linalg.det(cell)) * np.array(
                list(map(float, fields["stress"].split()))
            ).reshape(3, 3)
            if virial is not None and not np.allclose(virial, stress, atol=1e-5):
                raise ValueError("Inconsistent stress and virial")
            virial = stress
        if virial is not None:
            if not np.allclose(virial, virial.T, atol=1e-6):
                raise ValueError("Nonsymmetric virial")
            virial = virial.ravel()[[0, 4, 8, 1, 5, 6]] / n
        energy = float(fields["energy"]) / n
        weight = float(fields.get("weight", 1))
        values = np.r_[
            positions.ravel(),
            forces.ravel(),
            cell.ravel(),
            energy,
            weight,
            [] if virial is None else virial,
        ]
        if not np.isfinite(values).all() or weight <= 0:
            raise ValueError("Invalid XYZ numbers")
        frames.append(
            Frame(
                species,
                positions,
                cell,
                energy,
                forces,
                virial,
                weight,
                "".join(lines[start:stop]),
            )
        )
        start = stop
    if not frames:
        raise ValueError("Empty XYZ dataset")
    return frames


class Oracle:
    def __init__(self, bridge, model_path):
        self.model = Model.read(model_path)
        self.lib = C.CDLL(str(Path(bridge).resolve()))
        self.lib.fcvp_error.restype = C.c_char_p
        self.lib.fcvp_load.argtypes = [C.c_char_p]
        self.lib.fcvp_load.restype = C.c_void_p
        self.lib.fcvp_free.argtypes = [C.c_void_p]
        self.lib.fcvp_predict.argtypes = [
            C.c_void_p,
            C.c_int,
            IP,
            DP,
            DP,
            C.c_int,
            IP,
            DP,
            DP,
        ]
        self.lib.fcvp_solve.argtypes = [
            C.c_int,
            C.c_int,
            DP,
            DP,
            DP,
            DP,
            DP,
            C.c_int,
            C.c_double,
            C.c_double,
            C.c_int,
            C.c_double,
            C.c_double,
            C.c_double,
            C.c_double,
            DP,
            C.POINTER(C.c_int),
        ]
        self.handle = self.lib.fcvp_load(str(Path(model_path).resolve()).encode())
        if not self.handle:
            raise RuntimeError(self.lib.fcvp_error().decode())

    def close(self):
        if self.handle:
            self.lib.fcvp_free(self.handle)
            self.handle = None

    def predict(self, frame, beta):
        n = len(frame.species)
        types = np.array(
            [self.model.elements.index(e) for e in frame.species], np.int32
        )
        out = np.empty(1 + 3 * n + 6)
        ret = self.lib.fcvp_predict(
            self.handle,
            n,
            types,
            np.ascontiguousarray(frame.cell.T.ravel()),
            np.ascontiguousarray(frame.positions.T.ravel()),
            len(beta),
            self.model.indices,
            np.ascontiguousarray(beta, dtype=float),
            out,
        )
        if ret:
            raise RuntimeError(self.lib.fcvp_error().decode())
        return out

    def basis(self, frame):
        p = len(self.model.indices)
        beta = np.zeros(p)
        offset = self.predict(frame, beta)
        a = np.empty((len(offset), p))
        for j in range(p):
            beta[j] = 1
            a[:, j] = self.predict(frame, beta) - offset
            beta[j] = 0
        return a, offset

    def solve(
        self,
        groups,
        beta,
        iterations=40,
        epsilon=1e-8,
        tolerance=1e-8,
        lambda1=0.0,
        lambda2=0.0,
    ):
        h = np.ascontiguousarray([a.T @ a / len(y) for a, y, w in groups])
        g = np.ascontiguousarray([a.T @ y / len(y) for a, y, w in groups])
        yy = np.ascontiguousarray([y @ y / len(y) for a, y, w in groups])
        weights = np.ascontiguousarray([w for a, y, w in groups], dtype=float)
        fixed = self.model.fixed_regularization_parameters()
        x = np.ascontiguousarray(beta.copy())
        history = np.empty(iterations + 1)
        count = C.c_int()
        ret = self.lib.fcvp_solve(
            len(x),
            len(groups),
            h,
            g,
            yy,
            weights,
            x,
            iterations,
            epsilon,
            tolerance,
            self.model.training_parameter_count,
            lambda1,
            lambda2,
            float(np.abs(fixed).sum()),
            float(fixed @ fixed),
            history,
            C.byref(count),
        )
        if ret:
            raise RuntimeError(self.lib.fcvp_error().decode())
        return x, history[: count.value]


def groups_from_basis(
    frames,
    bases,
    weights=(1.0, 2.0, 0.1),
    type_weights=None,
    force_delta=0.0,
    shear=1.0,
):
    aa = [[], [], []]
    yy = [[], [], []]
    for frame, (a, offset) in zip(frames, bases):
        n = len(frame.species)
        scales = np.array(
            [
                1.0 if type_weights is None else type_weights.get(t, 1.0)
                for t in frame.species
            ]
        )
        if force_delta > 0:
            scales *= np.sqrt(
                force_delta / (force_delta + np.linalg.norm(frame.force, axis=1))
            )
        scales = np.tile(scales, 3) * frame.weight
        aa[0].append(a[:1] * frame.weight)
        yy[0].append((np.array([frame.energy]) - offset[:1]) * frame.weight)
        aa[1].append(a[1 : 1 + 3 * n] * scales[:, None])
        yy[1].append((frame.force.T.ravel() - offset[1 : 1 + 3 * n]) * scales)
        if frame.virial is not None:
            vs = frame.weight * np.array([1, 1, 1, shear, shear, shear])
            aa[2].append(a[-6:] * vs[:, None])
            yy[2].append((frame.virial - offset[-6:]) * vs)
    return [
        (np.concatenate(a), np.concatenate(y), w)
        for a, y, w in zip(aa, yy, weights)
        if a
    ]


def objective(
    groups, beta, epsilon=1e-8, lambda1=0.0, lambda2=0.0, fixed=None, total=None
):
    fixed = np.asarray([] if fixed is None else fixed)
    total = total or len(beta) + len(fixed)
    return (
        sum(
            w * np.sqrt(np.mean((a @ beta - y) ** 2) + epsilon**2) for a, y, w in groups
        )
        + lambda1 * (np.abs(beta).sum() + np.abs(fixed).sum()) / total
        + lambda2 * np.sqrt((beta @ beta + fixed @ fixed) / total + epsilon**2)
    )


def reference_mm(
    groups,
    beta,
    iterations=100,
    epsilon=1e-8,
    lambda1=0.0,
    lambda2=0.0,
    fixed=None,
    total=None,
):
    """Independent dense reference: rank-revealing SVD; L1 via a convex split.

    Unlike the production Gram solver, this never forms normal equations.
    """
    x = np.asarray(beta).copy()
    fixed = np.asarray([] if fixed is None else fixed)
    total = total or len(x) + len(fixed)
    opts = dict(
        epsilon=epsilon, lambda1=lambda1, lambda2=lambda2, fixed=fixed, total=total
    )
    history = [objective(groups, x, **opts)]
    for _ in range(iterations):
        matrices = []
        targets = []
        for a, y, w in groups:
            if w == 0:
                continue
            scale = np.sqrt(
                w / (len(y) * np.sqrt(np.mean((a @ x - y) ** 2) + epsilon**2))
            )
            matrices.append(scale * a)
            targets.append(scale * y)
        if lambda2:
            scale = np.sqrt(
                lambda2
                / (total * np.sqrt((x @ x + fixed @ fixed) / total + epsilon**2))
            )
            matrices.append(scale * np.eye(len(x)))
            targets.append(np.zeros(len(x)))
        if not matrices:
            break
        b = np.vstack(matrices)
        y = np.concatenate(targets)
        if lambda1:
            p = len(x)
            alpha = lambda1 / total

            def fun(z):
                v = z[:p] - z[p:]
                r = b @ v - y
                grad = b.T @ r
                return 0.5 * (r @ r) + alpha * z.sum(), np.r_[
                    grad + alpha, -grad + alpha
                ]

            fit = minimize(
                fun,
                np.r_[np.maximum(x, 0), np.maximum(-x, 0)],
                jac=True,
                bounds=[(0, None)] * (2 * p),
                method="L-BFGS-B",
                options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 20000},
            )
            candidate = fit.x[:p] - fit.x[p:]
        else:
            candidate = np.linalg.lstsq(b, y, rcond=1e-12)[0]
        value = objective(groups, candidate, **opts)
        for _ in range(40):
            if np.isfinite(value) and value <= history[-1]:
                break
            candidate = (candidate + x) * 0.5
            value = objective(groups, candidate, **opts)
        if not np.isfinite(value) or value > history[-1]:
            break
        change = np.max(np.abs(candidate - x))
        x = candidate
        history.append(value)
        if change < 1e-9 * max(1, np.max(np.abs(x))):
            break
    return x, np.array(history)
