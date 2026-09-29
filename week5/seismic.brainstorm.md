# `seismic` package sketch

The Rust binary in `week5/seismic/` follows
[`seismic.design.toml`](seismic.design.toml). Parse the experiment JSON with
`serde_json`; use flat row-major `[z][x]` arrays. Write the specified `.npy`
arrays directly. Pin the Enzyme nightly, isolate the numerical timestep in a
`no_std` static library, and call it through checked Rust wrappers.

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
- `born`: chain one Enzyme timestep JVP per update, seeding the velocity with
  `perturbation` and both initial pressure buffers with zero. Hold the source
  and damping fixed. Write scattered receiver samples to `born_data.npy`.
- `adjoint`: require `--data` pointing to a Born run's `born_data.npy`, with
  shape `[shot, step, receiver]`. Treat its samples as receiver weights. In
  full-storage mode, save the complete background state `(u[n-1], u[n])` at
  every `n = 0..steps`; Treeverse keeps only budgeted saved states and replays
  forward steps as needed. In either mode, inject receiver weights at the
  samples taken after each update and call Enzyme's timestep VJP once per
  reverse step. Carry the pressure adjoints through the state shift and sum
  the velocity adjoints into one `[z, x]` `f64` image. With Born data from the
  same perturbation, the image is `JᵀJ*perturbation`.

`--every k` applies only to forward and adjoint modes. Count the interval
from step 0 without recording a step-0 frame. Forward records the first shot
after updates `k, 2k, ...` through the largest multiple of `k` at most `N`.
Adjoint records those same positive multiples in decreasing order. A frame at
step `n` holds the pressure adjoint of `u[n]` after injecting that step's
receiver weights and before the timestep VJP.
Store frame steps and times `step*dt` in `run.json`, in recording order.
Reject `--every` in Born mode. Process shots one at a time. Full
storage uses `steps+1` complete states per shot; each state holds two `f64`
fields and occupies `2*nx*nz*8` bytes. The 216,545-cell, 1,200-step Marmousi
case therefore needs about 4.16 GB for saved states of one shot.

For adjoint `result.json`, report `storage: "full"`, `checkpoints: null`,
`reverse_calls`, `scheduler_forward_calls`, `peak_saved_states`, and
`peak_saved_bytes`. Report the calls and peak state count per shot in shot
order; total the calls across shots and report the largest peak. With full
storage, each shot takes `steps` forward calls to populate its `steps+1`
states and `steps` VJP calls to reverse them. The forward step evaluated
inside a VJP is not a scheduler forward call.

With `--storage treeverse --checkpoints d`, keep state 0 and at most `d`
additional complete states. Choose the smallest positive `t` with
`binomial(d+t, d) >= steps`, then run the specified recursive `visit` schedule.
Restore saved bases and replay forward calls to create each needed state;
apply the same timestep VJP and receiver injection at each `grad` action.
Write `actions-<shot>.json` with the store, restore, call, grad, and fetch
actions and their saved-state counts. Count replayed `call` actions as
`scheduler_forward_calls`; include each action file in the shot's statistics.

## Checks that matter

Validate array dimensions, positive speeds and spacings, in-grid survey
points, and the two-dimensional CFL bound `max(c)*dt/dx <= 1/sqrt(2)` before
running. The supplied reflector and Marmousi backgrounds have CFL numbers
`0.36` and `0.517`, respectively. Check the forward step and receiver timing
on the reflector input; compare Born data with a centered perturbation of the
forward model; check `⟨Jδc, d⟩ = ⟨δc, Jᵀd⟩` with independently chosen receiver
weights `d`, then check the Born-data image against `JᵀJδc`. Keep `inputs/`
and generated `.npy` files out of Git.
