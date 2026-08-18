# Shared extension primitives

- `sampling/gaussian_position.cuh`: XY Gaussian + wrap into the box
- `sampling/gaussian_velocity.cuh`: fixed-|v| beam, Gaussian theta, uniform phi, along -z
- `surface/local_surface.cuh`: local (or global fallback) zmax
- `topology/connectivity_bfs.cuh`: keep-mask for atoms connected to the lowest-z atom
