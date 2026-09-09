#!/usr/bin/env python3
import math
import os
import sys

PI = 3.14159265358979


def fibonacci_directions(M):
    golden = PI * (3.0 - math.sqrt(5.0))
    dirs = []
    for m in range(M):
        z = 1.0 - 2.0 * (m + 0.5) / M
        xy = math.sqrt(max(0.0, 1.0 - z * z))
        theta = m * golden
        dirs.append((xy * math.cos(theta), xy * math.sin(theta), z))
    return dirs


def mic(dx, dy, dz, L):
    dx -= L * round(dx / L)
    dy -= L * round(dy / L)
    dz -= L * round(dz / L)
    return dx, dy, dz


def ball_restricted_volume(i, positions, L, R, directions):
    inv_R = 1.0 / R
    cutoff2 = 4.0 * R * R
    xi, yi, zi = positions[i]
    q = [inv_R] * len(directions)
    for j, (xj, yj, zj) in enumerate(positions):
        if j == i:
            continue
        dx, dy, dz = mic(xj - xi, yj - yi, zj - zi, L)
        d2 = dx * dx + dy * dy + dz * dz
        if d2 <= 0.0 or d2 >= cutoff2:
            continue
        scale = 2.0 / d2
        bx, by, bz = scale * dx, scale * dy, scale * dz
        for m, (nx, ny, nz) in enumerate(directions):
            q[m] = max(q[m], bx * nx + by * ny + bz * nz)
    acc = 0.0
    for qi in q:
        radius = 1.0 / qi
        acc += radius * radius * radius
    return (4.0 * PI / (3.0 * len(directions))) * acc


def read_extxyz(path):
    with open(path) as f:
        n = int(f.readline())
        header = f.readline()
        rows = [f.readline().split() for _ in range(n)]
    volumes = [float(row[-1]) for row in rows]
    positions = [(float(row[1]), float(row[2]), float(row[3])) for row in rows]
    return header, positions, volumes


def check_isolated(path):
    header, _, volumes = read_extxyz(path)
    if "voronoi_method=\"ball_restricted\"" not in header:
        sys.exit("isolated: missing voronoi_method in header")
    if "volume_atom:R:1" not in header:
        sys.exit("isolated: missing volume_atom property")
    if len(volumes) != 2:
        sys.exit(f"isolated: expected 2 atoms, got {len(volumes)}")
    R = 3.0
    expected = 4.0 * PI * R**3 / 3.0
    for i, volume in enumerate(volumes):
        err = abs(volume - expected) / expected
        if err > 1e-9:
            sys.exit(f"isolated: atom {i} volume {volume} != 4/3 pi R^3 = {expected}, relerr={err}")
    print(f"isolated: PASS volume={volumes[0]:.12g}")


def check_simple_cubic(path):
    header, positions, volumes = read_extxyz(path)
    a = 4.0
    nside = 4
    L = a * nside
    R = 3.5
    M = 128
    if len(volumes) != nside**3:
        sys.exit(f"simple_cubic: expected {nside**3} atoms, got {len(volumes)}")
    dirs = fibonacci_directions(M)
    ref = [ball_restricted_volume(i, positions, L, R, dirs) for i in range(len(positions))]
    max_kernel_err = max(abs(v - r) / r for v, r in zip(volumes, ref))
    if max_kernel_err > 1e-9:
        sys.exit(f"simple_cubic: GPU disagrees with CPU quadrature, max relerr={max_kernel_err}")
    mean = sum(volumes) / len(volumes)
    rel_bulk = abs(mean - a**3) / a**3
    if rel_bulk > 0.05:
        sys.exit(f"simple_cubic: mean volume {mean} not within 5% of a^3={a**3}")
    print(
        f"simple_cubic: PASS mean={mean:.6f} a^3={a**3:.6f} "
        f"quad_relerr={rel_bulk:.4e} kernel_relerr={max_kernel_err:.3e}"
    )


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    check_isolated(os.path.join(root, "isolated", "vol.xyz"))
    check_simple_cubic(os.path.join(root, "simple_cubic", "vol.xyz"))


if __name__ == "__main__":
    main()
