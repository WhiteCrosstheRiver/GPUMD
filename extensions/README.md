# GPUMD extensions

`src/` is the official GPUMD core. `extensions/` holds deposition-specific algorithms. Samplers, surface query, and connectivity BFS are header-only and included from `deposit.cu` / `delete.cu`.

## Command vs algorithm

```text
for                 control flow (in src)
deposit             create atoms (thin command in src)
delete              remove atoms (thin command in src)
Gaussian / surface  give deposit a position and velocity
BFS connectivity    give delete a mask; default off
```

`AtomMutation` is an internal helper, not a subsystem. After CPU atom arrays change, one contract restores N-dependent GPU/Group/Force state: `rebuild_after_mutation`. The class stays for now; do not grow it into a framework.

Do not port the old Python dump→parse→BFS→write-xyz→restart loop. Use:

```text
for i range 1 500
    deposit Ge position gaussian 0 0 8.33 velocity gaussian 0.007 5 surface local 5.0 offset antivel 2.6
    run 100
end
```

`delete disconnected` is explicit. `deposit` must not run BFS. BFS returns a keep mask only; it does not call `AtomMutation`.

## Layout

```text
extensions/common/
  sampling/     gaussian_position.cuh, gaussian_velocity.cuh
  surface/      local_surface.cuh
  topology/     connectivity_bfs.cuh
```

## Dynamic N

`GPU_Vector::resize` frees then mallocs. Never change `N` by resizing GPU arrays in place.

CPU arrays are the source of truth. `rebuild_after_mutation` rebuilds GPU Atom/Group buffers and calls `Force::update_number_of_atoms` (NEP `ensure_memory_for_atoms`).

Not supported yet: changing `N` while PIMD beads exist.

`electron_stop` and `add_random_force` size some buffers at parse time. Issue those keywords **after** deposit/delete in the same run block, or they will not see the new `N`.
