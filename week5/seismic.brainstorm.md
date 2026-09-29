# `seismic` package sketch

Build a Rust binary in `week5/seismic/` with the CLI and output files in
[`seismic.design.toml`](seismic.design.toml). Parse the experiment JSON with
`serde_json`; use flat row-major `[z][x]` arrays. A direct `.npy` writer can
emit the three specified array shapes without an array framework.

## Discrete step

For interior cell `(x, z)`, let `d = min(x, z, nx-1-x, nz-1-z)` and
`sigma = sponge_strength * max(0, 1-d/sponge_width)^2`. With the five-point
Laplacian `L` and source at `t_n = n*dt`, the centered update is

```text
u[n+1] = (2*u[n] - (1-sigma*dt)*u[n-1]
          + dt^2*(c^2*L(u[n]) + q[n])) / (1+sigma*dt)
```

Initialize both pressure buffers to zero. Update only interior cells, leaving
the outer row and column at zero. A Ricker pulse uses
`s = pi*f0*(t_n-t0)` and `source_amplitude*(1-2*s^2)*exp(-s^2)`. Its spatial
footprint is `exp(-((x-shot_x)^2+(z-shot_z)^2)/2)`, which peaks at one for the
integer-centered shots in both supplied experiments. Sample receivers from
`u[n+1]`, so trace index zero is the field after the first update.

## Modes

- `forward`: run the update for each shot and write pressure traces as
  `[shot, step, receiver]` `f64`.
- `born`: advance the background and scattered fields together. The scattered
  source is `2*c*perturbation*L(u[n])`; its initial buffers and boundaries are
  zero. Write the scattered receiver samples as `born_data.npy`.
- `adjoint`: use the internally generated Born traces as receiver data and
  apply the **discrete transpose** of the Born update to form a velocity image
  `[z, x]` `f64`. The spatial transpose matters: for varying `c`, the transpose
  of `c^2*L` is `L(c^2*·)`. This image is `JᵀJ*perturbation`, where `J` maps a
  velocity perturbation to scattered traces.

For forward `--recording-every k`, record the first shot after updates
`k, 2k, ...`; adjoint recording follows decreasing steps as specified in the
TOML. Store frame steps and times `step*dt` in `run.json`. Process shots one
at a time. The Marmousi grid has 216,545 cells; a full `f64` wavefield history
for one 1,200-step shot is about 2.1 GB, so the adjoint needs file-backed
forward history or bounded recomputation rather than retaining all frames in
RAM.

## Checks that matter

Validate array dimensions, positive speeds and spacings, in-grid survey
points, and the two-dimensional CFL bound `max(c)*dt/dx <= 1/sqrt(2)` before
running. The supplied reflector and Marmousi backgrounds have CFL numbers
`0.36` and `0.517`, respectively. Check the forward step and receiver timing
on the reflector input; compare Born data with a centered perturbation of the
forward model; check the adjoint with a dot-product identity. Keep `inputs/`
and generated `.npy` files out of Git.
