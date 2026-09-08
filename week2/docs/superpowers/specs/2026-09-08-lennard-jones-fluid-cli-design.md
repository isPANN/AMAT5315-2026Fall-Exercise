# Lennard–Jones fluid CLI design

## Goal and commands

Extend the Rust crate at `week2/md` into a two-dimensional Lennard–Jones fluid
tool with three subcommands sharing one trajectory format:

```sh
md run --n 100 --rho 0.8 --temperature 0.5 --dt 0.01 \
  --eq-steps 2000 --steps 10000 --sample-every 50 --seed 2026 --out artifacts
md run --out artifacts
md check artifacts
md video artifacts --out artifacts/run.mp4
```

The first two commands are equivalent. All numerical run options default to
the explicit values above. Require `--out` for run and video; the input directory
is positional for check and video. The executable is named `md`.

Add `week2/Makefile`; running `make reproduce` from `week2/` executes the default
run in release mode and writes `week2/artifacts`. Document the Cargo equivalents
and the three commands in the week README. A global installation is not required
by the reproduction target.

## Physical model and geometry

Use two dimensions and reduced units with atom mass, epsilon, sigma, and
Boltzmann constant all equal to 1. Let `q = sqrt(n)`. Require q to be an even
integer so the alternating row offset joins consistently across the periodic
y boundary. The prescribed 100, 400, and 1600 atom cases use 10, 20, and 40 rows
and columns.

At number density rho, define

```text
a = sqrt(2 / (sqrt(3) * rho))
h = sqrt(3) * a / 2
Lx = q * a
Ly = q * h
position(i, j) = [(i + 0.5 * (j mod 2)) * a, j * h]
```

Enumerate rows j first, then columns i, both from 0 through q-1. The contract
box is approximately `[12.0141, 10.4045]`; compute it from the formulas rather
than using rounded constants. At fixed density, a and h remain unchanged when
the atom count changes.

Wrap each position component into `[0, L)` after stepping. Evaluate pair
displacements using the minimum image, `d -= L * round(d / L)` independently
on both axes. Require both box sides to exceed `2 * rc`, where `rc = 2.5`.

Use the existing scalar Lennard–Jones functions:

```text
U(r) = 4 * (r^(-12) - r^(-6))
f(r) = 24 * (2 * r^(-13) - r^(-7))
Us(r) = U(r) - U(rc) for r < rc; otherwise 0
```

Inside the cutoff the force remains the ordinary Lennard–Jones force; this is
a potential shift, not a force shift. For minimum-image displacement d from i
to j, the force on j is `f(r) * d / r` and on i is its negative. At or beyond
the cutoff both interaction energy and force are zero. Count each pair's
potential once. No tail correction, force softening, or force clamping is used.

Start with a direct loop over unordered pairs. This supports the specified atom
counts with quadratic work per force evaluation. Add cell lists only if measured
runtime later establishes that they are needed; no cell-list infrastructure is
part of this change.

## Initialization, equilibration, and production

Use a deterministic pseudorandom stream seeded by `seed`. Draw independent
Gaussian velocity components with mean zero and variance equal to the configured
target temperature (0.5 for the contract run), in atom order, x then y. Subtract
the vector mean velocity from every atom once to remove centre-of-mass motion.

Compute `E_kin = 0.5 * sum_i |v_i|^2` and
`T_thermo = 2 * E_kin / (2*n - 2)`. Multiply every velocity component by
`sqrt(temperature / T_thermo)`. Apply this rescaling initially and immediately
after equilibration steps 50, 100, 150, and so on. Do not add an extra rescaling
at an equilibration endpoint that is not divisible by 50.

Advance using velocity-Verlet: move using old velocity and acceleration,
recompute forces at the new positions, then update velocities using the average
of old and new acceleration. The fluid CLI exposes this integrator only.

After `eq_steps`, reset the production step counter and time to zero without
resetting the physical state. Run `steps` more integration steps with no
thermostat. Save a frame only when the positive production step is divisible by
`sample_every`. Do not save production step zero or an additional unscheduled
final frame. The number of saved frames is `floor(steps / sample_every)`;
the contract run has 200 frames and the 1000/10 example has exactly 100.

## File contract

`run` creates the output directory and writes `run.json` and `traj.jsonl` there.
Rerunning in the same directory replaces those two outputs without deleting
other files in the directory. Write full floating-point serialization precision.

`run.json` is one JSON object containing:

| Field | Value/type |
| --- | --- |
| `n` | Integer atom count |
| `rho` | Number density |
| `box` | `[Lx, Ly]`, computed from n and rho |
| `dt` | Integration time step |
| `temperature` | Target temperature |
| `eq_steps` | Integer equilibration step count |
| `steps` | Integer production step count |
| `sample_every` | Integer production sampling interval |
| `seed` | Integer random seed |
| `integrator` | Exactly `"velocity-verlet"` |

`traj.jsonl` contains one JSON object per line, in ascending production-step
order. Each object has `step`, `t`, `pos`, `vel`, `E_pot`, and `E_kin`:

- `step` is the positive production step; `t = step * dt`.
- `pos` and `vel` are arrays of n two-component arrays, in unchanged atom order.
- Positions are wrapped into the box; velocities are not wrapped.
- `E_pot` uses the potential-shifted cutoff above; `E_kin` includes all atoms.

The trajectory is the sole state source for check and video. Neither command
reruns equilibration, generates velocities, or advances the integrator.

## Input and file validation

At the CLI and saved-file boundaries, reject invalid integer counts, a grid
that violates the geometry requirements, nonfinite values, nonpositive rho,
temperature or dt, zero sampling intervals, and runs with no saved frames.
Allow zero equilibration steps. Reject unknown command-line options.

Both readers validate required fields and types, supported integrator identity,
box consistency with n and rho, expected frame count and sampling sequence,
time consistency, particle counts, two-component vectors, finite numbers, and
wrapped coordinate ranges. Allow ordinary floating-point roundoff in computed
box/time comparisons, without changing stored values. Coincident particles
and nonfinite recomputed physical quantities are explicit errors.

Stored energies must be present and finite, but the physical measurements use
energies recomputed from positions and velocities, never those stored numbers.
Do not add a fourth physical acceptance gate comparing stored and recomputed
energy. The three required gates below determine physical PASS/FAIL.

Errors identify the invalid input or affected frame. Propagate parsing, file,
simulation, rendering, and encoder failures with unsuccessful process exit
status. Do not silently skip samples, repair malformed files, or substitute
default data.

## Physics checks

Recompute `E_pot`, `E_kin`, and `E = E_pot + E_kin` for every saved frame.
Let `E0` be the recomputed total energy of the first saved frame and
`k = max(1, floor(frame_count / 10))`. The secular drift is

```text
abs(mean(E over last k frames) - mean(E over first k frames)) / abs(E0)
```

Zero reference energy makes this statistic undefined and is an explicit error.
Do not normalize against the start of equilibration or an unsaved production
step zero.

Pool all speeds from saved production frames. With M pooled speeds, compute
`T_speed = mean(v^2) / 2`. This differs intentionally from the thermostat's
finite-system temperature. The predicted two-dimensional speed density is
`f(v) = (v / T_speed) * exp(-v^2 / (2*T_speed))`.
Zero pooled speed temperature makes the shape statistic undefined and is an
explicit error.

Use exactly 24 equal-predicted-probability bins. Their edges are
`b_k = sqrt(-2*T_speed*ln(1-k/24))` for k from 0 through 23, with b_24 infinite.
Assign speeds to half-open intervals `[b_k, b_(k+1))`. Each expected count is
`M / 24`, using floating-point division. Compute

```text
reduced_chi_squared = sum_bins((observed - expected)^2 / expected) / 22
```

Print these rows with each measured value beside its strict limit and PASS/FAIL:

| Check | Measured quantity | Pass condition |
| --- | --- | --- |
| Secular drift | Drift defined above | `< 0.002` |
| Temperature | `abs(T_speed - temperature)` | `< 0.05` |
| Speed shape | Reduced chi-squared defined above | `< 2` |

Also print T_speed itself so the temperature result is interpretable. The target
is the temperature in metadata, so it is exactly 0.5 for the contract run and
the configured value for a non-default run. If any physical gate fails, print
all three results and exit unsuccessfully. The shape threshold is a practical
tolerance for correlated samples, not a formal significance test.

## Pair structure and video

Use exactly 100 equal-width radial bins from zero to
`r_max = min(Lx, Ly) / 2`. Count minimum-image pair distances below r_max, using
half-open bins; each unordered pair contributes two neighbour counts. At video
frame f, accumulate counts over saved frames 1 through f and divide each bin by

```text
f * n * rho * pi * (outer_radius^2 - inner_radius^2)
```

Use the specified rho normalization, without a finite-N correction. Plot g(r)
at bin centres. The displayed long-range contrast is
`sqrt(mean((g(r)-1)^2))` over bins whose centres satisfy `r > 2`. It is computed
from the displayed cumulative RDF, not averaged from separate frame contrasts.

Render two side-by-side panels: atoms at the current saved wrapped positions
and the cumulative RDF. Preserve the physical box aspect ratio. Label radius,
g(r), production time, and long-range contrast; show the uniform-density
reference g(r)=1. Use stable axes across the video so curve changes reflect
data rather than changing scales, and avoid clipping peaks.

Use the existing Plotters dependency for rendering and the local ffmpeg
executable for MP4 encoding. ffmpeg and ffprobe were found locally during design.
Document ffmpeg as a prerequisite for video generation on another machine.

Encode one frame per saved frame, in order, at 30 frames per second. The contract
video therefore has 200 frames and lasts about 6.67 seconds. Use an encoding
bitrate budget for the clip duration and verify the finished MP4 is strictly
smaller than 2,000,000 bytes. Do not drop frames or truncate the trajectory to
meet the size limit. Encoding or size-limit failures are explicit errors.

## Code organization and repository scope

Reuse the existing scalar pair physics and evolve the current state and
integrator implementation directly. Separate CLI parsing, simulation,
trajectory I/O, analysis, and rendering only as needed for readable Rust files.
Use typed JSON serialization and an established seeded random-number library;
keep existing Plotters support. Do not write a custom JSON parser or random
number generator. Select concrete dependency versions during implementation
planning.

The current working tree has an untracked week README, dimer example, and
dimer image. Preserve that work. Update affected dimer example calls if the
shared library types change; keep it using the same physics implementation,
without copying or freezing the old library. Its open-boundary dimer physics
must not accidentally acquire the fluid cutoff or periodic wrapping.

Replace the binary's fixed dimer-output workflow with the three subcommands.
Do not introduce general force-model traits, registries, compatibility wrappers,
versioned implementations, or a second simulation engine. Keep changes under
week2 except where an existing repository convention explicitly requires
otherwise. Exclude generated trajectories, videos, rendered frames, and build
artifacts from commits. Stage only changes required by this task.

## Verification and acceptance

Use focused runnable Rust checks for minimum-image pair forces, shifted energy
and zero interaction at the cutoff, equal-and-opposite forces, lattice geometry,
thermostat degrees of freedom and zero mean velocity, Verlet stepping, and
production sampling without step zero. Verify repeatability with the same seed
using semantic comparisons rather than committed trajectory snapshots.

Use small hand-checkable saved states to verify independently recomputed energy,
early/late averaging, pooled speed temperature, the 24-bin statistic, RDF
normalization, and long-range contrast. Exercise malformed input and failed
physics gates, including unsuccessful CLI exit status. Checks must consume saved
files rather than trusting reported simulation diagnostics.

Run formatting and Cargo tests, then `make reproduce` and `md check` against the
full default run. All three contract gates must pass without altering the seed,
parameters, cutoff model, sampling, or limits. Record the actual measured values
in the completion report. Any failure requires diagnosis rather than weakening
the acceptance criteria.

Generate the contract video, verify its decoded frame count equals 200 and its
size is below the limit, and inspect early and late rendered frames for readable
labels, correct box geometry, particle visibility, and cumulative RDF behavior.
Confirm the generated artifacts are ignored and the final Git diff is limited
to the requested tool. No numerical or video acceptance results are claimed by
this specification; those checks occur during implementation.
