#!/usr/bin/env python3
"""Minimal CPU UF3 evaluator matching GPUMD force/uf3.cu conventions."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field


def eval_cubic(c, u):
    a, b, cc, d = c
    return a + u * (b + u * (cc + u * d))


def eval_cubic_deriv(c, u, inv_h):
    a, b, cc, d = c
    return (b + u * (2 * cc + 3 * d * u)) * inv_h


def bspline4(u):
    u2, u3 = u * u, u * u * u
    i6 = 1 / 6
    return np.array(
        [(1 - 3 * u + 3 * u2 - u3) * i6, (4 - 6 * u2 + 3 * u3) * i6,
         (1 + 3 * u + 3 * u2 - 3 * u3) * i6, u3 * i6]
    )


def bspline4_deriv(u, inv_h):
    om, u2 = 1 - u, u * u
    return np.array([-om * om * 0.5, u * (3 * u - 4) * 0.5,
                     (-3 * u2 + 2 * u + 1) * 0.5, u2 * 0.5]) * inv_h


def precompute_2b_uniform(knots, coeffs):
    nk, nc = len(knots), len(coeffs)
    nint = nk - 1
    table = []
    for m in range(nint):
        i0 = min(max(0, m - 3), nc - 1)
        i1 = min(max(0, m - 2), nc - 1)
        i2 = min(max(0, m - 1), nc - 1)
        i3 = min(max(0, m), nc - 1)
        c0, c1, c2, c3 = coeffs[i0], coeffs[i1], coeffs[i2], coeffs[i3]
        table.append(
            ((c0 + 4 * c1 + c2) / 6, (-3 * c0 + 3 * c2) / 6,
             (3 * c0 - 6 * c1 + 3 * c2) / 6, (-c0 + 3 * c1 - 3 * c2 + c3) / 6)
        )
    km = knots[0]
    kd = (knots[-1] - km) / nint
    return table, km, kd, 1.0 / kd, nint


@dataclass
class UF3Model:
    elements: list = field(default_factory=list)
    e0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    coeff_2b: dict = field(default_factory=dict)
    tensor_3b: dict = field(default_factory=dict)
    rc_max: float = 0.0
    rc3: tuple = (0.0, 0.0, 0.0)
    knots3: tuple = (None, None, None)
    nc3: tuple = (0, 0, 0)
    nint3: tuple = (0, 0, 0)
    skin: float = 1.0

    def type_of(self, sym):
        return self.elements.index(sym)

    @classmethod
    def load(cls, path):
        lines = [ln.strip() for ln in Path(path).read_text().splitlines()
                 if ln.strip() and not ln.startswith('#')]
        m = cls()
        i = 0
        while i < len(lines):
            tok = lines[i].split()
            if tok[0] == 'uf3':
                m.elements = tok[2:2 + int(tok[1])]
                m.e0 = np.zeros(len(m.elements))
                i += 1
                continue
            if tok[0] == '1B':
                m.e0 = np.array(list(map(float, tok[1:])), dtype=np.float64)
                i += 1
                continue
            if tok[0] == '2B':
                e1, e2 = tok[1], tok[2]
                rc = float(lines[i + 1].split()[0])
                nk = int(lines[i + 1].split()[1])
                knots = list(map(float, lines[i + 2].split()))
                ncoeff = int(lines[i + 3].split()[0])
                coeffs = list(map(float, lines[i + 4].split()))
                table, km, kd, inv_kd, nint = precompute_2b_uniform(knots, coeffs)
                m.coeff_2b[(e1, e2)] = {
                    'rc': rc, 'table': table, 'kmin': km, 'kd': kd,
                    'inv_kd': inv_kd, 'nint': nint,
                }
                m.rc_max = max(m.rc_max, rc)
                i += 5
                continue
            if tok[0] == '3B':
                e1, e2, e3 = tok[1], tok[2], tok[3]
                p = lines[i + 1].split()
                rc_ij, rc_ik, rc_jk = map(float, p[:3])
                nk_ij, nk_ik, nk_jk = map(int, p[3:6])
                k_ij = list(map(float, lines[i + 2].split()))
                k_ik = list(map(float, lines[i + 3].split()))
                k_jk = list(map(float, lines[i + 4].split()))
                nc_ij, nc_ik, nc_jk = map(int, lines[i + 5].split())
                tensor = np.zeros((nc_ij, nc_ik, nc_jk))
                row = 0
                for pi in range(nc_ij):
                    for qi in range(nc_ik):
                        vals = list(map(float, lines[i + 6 + row].split()))
                        tensor[pi, qi, :len(vals)] = vals
                        row += 1
                m.tensor_3b[(e1, e2, e3)] = {'tensor': tensor}
                m.rc3 = (rc_ij, rc_ik, rc_jk)
                m.knots3 = (k_ij, k_ik, k_jk)
                m.nc3 = (nc_ij, nc_ik, nc_jk)
                m.nint3 = (nk_ij - 1, nk_ik - 1, nk_jk - 1)
                m.rc_max = max(m.rc_max, rc_ij, rc_ik, rc_jk)
                i += 6 + nc_ij * nc_ik
                continue
            i += 1
        return m

    def eval(self, species, pos, box=None):
        """Static energy/forces using full periodic images (ghost-supercell
        convention), matching GPUMD's multi-image UF3 kernels.  When the cell is
        larger than 2*rc this reduces to the minimum-image result; when it is
        smaller, every image within rc is summed (minimum image alone is wrong).
        The asymmetric-cutoff 3-body term is averaged over both neighbour
        orderings (dual_order) so the result is neighbour-order independent."""
        pos = np.asarray(pos, dtype=np.float64)
        n = len(species)
        types = [self.type_of(s) for s in species]
        E = float(np.sum(self.e0[types]))
        F = np.zeros((n, 3))
        rc_nl = self.rc_max

        # Periodic image range per lattice direction (columns of box are a,b,c).
        if box is not None:
            a, b, c = box[:, 0], box[:, 1], box[:, 2]
            vol = abs(np.dot(a, np.cross(b, c)))
            def perp(u, v):
                return vol / max(np.linalg.norm(np.cross(u, v)), 1e-6)
            P = [int(np.ceil(rc_nl / perp(b, c))),
                 int(np.ceil(rc_nl / perp(a, c))),
                 int(np.ceil(rc_nl / perp(a, b)))]
        else:
            a = b = c = np.zeros(3)
            P = [0, 0, 0]

        # Multi-image neighbour list: per centre i, all (j, image_position) with
        # 0 < |image - i| < rc_nl.  Stored as absolute positions so the 3-body
        # triangle (r12,r13,r23) is self-consistent.
        def neighbors(i):
            out = []
            for j in range(n):
                for i1 in range(-P[0], P[0] + 1):
                    for i2 in range(-P[1], P[1] + 1):
                        for i3 in range(-P[2], P[2] + 1):
                            if j == i and i1 == 0 and i2 == 0 and i3 == 0:
                                continue
                            img = pos[j] + i1 * a + i2 * b + i3 * c
                            d2 = float(np.dot(img - pos[i], img - pos[i]))
                            if 1e-12 < d2 < rc_nl * rc_nl:
                                out.append((j, img))
            return out

        nbrs_all = [neighbors(i) for i in range(n)]

        # 2B — directed NL, 0.5 energy, full force on centre i.
        for i in range(n):
            for (j, img) in nbrs_all[i]:
                d = self.coeff_2b.get((species[i], species[j]))
                if d is None:
                    continue
                dx = img - pos[i]
                r = np.linalg.norm(dx)
                if r >= d['rc'] or r < 1e-12:
                    continue
                mi = int((r - d['kmin']) * d['inv_kd'])
                mi = max(0, min(mi, d['nint'] - 1))
                u = (r - d['kmin'] - mi * d['kd']) * d['inv_kd']
                E += 0.5 * eval_cubic(d['table'][mi], u)
                F[i] += (eval_cubic_deriv(d['table'][mi], u, d['inv_kd']) / r) * dx

        rc_ij, rc_ik, rc_jk = self.rc3
        k_ij, k_ik, k_jk = self.knots3
        nc_ij, nc_ik, nc_jk = self.nc3
        nint_ij, nint_ik, nint_jk = self.nint3

        def leg(r, knots, nint):
            km = knots[0]
            kd = (knots[-1] - km) / nint
            inv_kd = 1.0 / kd
            mi = int((r - km) * inv_kd)
            mi = max(0, min(mi, nint - 1))
            u = (r - km - mi * kd) * inv_kd
            return bspline4(u), bspline4_deriv(u, inv_kd), max(0, mi - 3)

        # One ordered triplet (centre i; pa on ij leg, pb on ik leg) scaled by w.
        # Returns energy and the three force legs (on i, on pa-atom, on pb-atom).
        def triplet(ti, ta, tb, pi, pa, pb, w):
            x12 = pa - pi; r12 = np.linalg.norm(x12)
            if r12 >= rc_ij or r12 < 1e-12:
                return 0.0, None
            x13 = pb - pi; r13 = np.linalg.norm(x13)
            if r13 >= rc_ik or r13 < 1e-12:
                return 0.0, None
            x23 = pb - pa; r23 = np.linalg.norm(x23)
            if r23 >= rc_jk or r23 < 1e-12:
                return 0.0, None
            T = self.tensor_3b.get((ti, ta, tb))
            if T is None:
                return 0.0, None
            tensor = T['tensor']
            bij, dbij, p0 = leg(r12, k_ij, nint_ij)
            bik, dbik, q0 = leg(r13, k_ik, nint_ik)
            bjk, dbjk, r0 = leg(r23, k_jk, nint_jk)
            val = dv12 = dv13 = dv23 = 0.0
            for dp in range(4):
                p = p0 + dp
                if p >= nc_ij:
                    break
                for dq in range(4):
                    q = q0 + dq
                    if q >= nc_ik:
                        break
                    bpbq = bij[dp] * bik[dq]
                    dbpbq = dbij[dp] * bik[dq]
                    bpdbq = bij[dp] * dbik[dq]
                    Rv = Rd23 = 0.0
                    for dr in range(4):
                        rr = r0 + dr
                        if rr >= nc_jk:
                            break
                        C = tensor[p, q, rr]
                        Rv += C * bjk[dr]
                        Rd23 += C * dbjk[dr]
                    val += bpbq * Rv
                    dv12 += dbpbq * Rv
                    dv13 += bpdbq * Rv
                    dv23 += bpbq * Rd23
            t12, t13, t23 = w * dv12 / r12, w * dv13 / r13, w * dv23 / r23
            a12, a13, a23 = t12 * x12, t13 * x13, t23 * x23
            f_i = a12 + a13
            f_a = a23 - a12
            f_b = -a13 - a23
            return w * val, (f_i, f_a, f_b)

        # 3B — unordered neighbour pairs, dual-order averaged (w = 0.5 each)
        # so energies/forces do not depend on the neighbour ordering even when
        # the ij and ik cutoffs/grids differ.
        for i in range(n):
            ti = species[i]
            nb = nbrs_all[i]
            for jj in range(len(nb)):
                ja, pa = nb[jj]
                for kk in range(jj + 1, len(nb)):
                    kb, pb = nb[kk]
                    # assignment A: pa on ij leg, pb on ik leg
                    eA, fA = triplet(ti, species[ja], species[kb], pos[i], pa, pb, 0.5)
                    if fA is not None:
                        E += eA
                        F[i] += fA[0]; F[ja] += fA[1]; F[kb] += fA[2]
                    # assignment B: pb on ij leg, pa on ik leg
                    eB, fB = triplet(ti, species[kb], species[ja], pos[i], pb, pa, 0.5)
                    if fB is not None:
                        E += eB
                        F[i] += fB[0]; F[kb] += fB[1]; F[ja] += fB[2]
        return E, F
