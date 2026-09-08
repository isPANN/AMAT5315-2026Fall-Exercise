# Two-atom molecular dynamics design

## Purpose

Extend `week2/md` to compare forward Euler and velocity-Verlet through one
Rust integrator trait and one experiment runner. The experiment outputs total
energy and signed total-energy error as CSV, plus a comparison plot.

## Physical model

- Two atoms in two dimensions, each with mass 1.
- Reduced Lennard–Jones units: epsilon and sigma are both 1.
- Open boundaries; no walls, periodic wrapping, cutoff, or potential shifting.
- Initial positions: atom i at `(0.0, 0.0)`, atom j at `(1.2, 0.0)`.
- Both initial velocities are `(0.0, 0.0)`.

Reuse the existing functions for
`U(r) = 4 * (r^(-12) - r^(-6))` and radial force magnitude
`f(r) = 24 * (2*r^(-13) - r^(-7))`.
With displacement `d = position_j - position_i` and `r = |d|`,
the force on j is `f(r) * d/r`, and the force on i is its negative.
Accelerations equal forces because masses are 1.

The state is genuinely two-dimensional, although these initial conditions
produce motion along the x-axis. Do not add transverse motion.

## State and integrator interface

Keep the physics, integrators, and experiment runner in `week2/md/src/lib.rs`.
Represent state directly as two positions and two velocities, each using
`[[f64; 2]; 2]`. Use one public `Integrator` trait with the method
`fn step(&self, state: &mut State, dt: f64)` and two concrete implementations,
`ForwardEuler` and `VelocityVerlet`.

Forward Euler evaluates acceleration from the old positions and updates
`position_new = position_old + dt * velocity_old` and
`velocity_new = velocity_old + dt * acceleration_old`.
Both updates must use old state values; updating velocity first and using it
for the position would implement a different method.

Velocity-Verlet evaluates the old acceleration, updates positions using
`position_new = position_old + dt * velocity_old + 0.5 * dt^2 * acceleration_old`,
then evaluates acceleration at the new positions and updates velocities using
`velocity_new = velocity_old + 0.5 * dt * (acceleration_old + acceleration_new)`.
Recompute accelerations directly; no cached-force machinery is needed.

One runner accepts an integrator, initial state, time step, and step count.
It records the initial state and the state after each completed step.
Compute sample time as `step as f64 * dt`.

## Experiment and measurement

Use a fixed time step of `0.01` for both methods, starting each from the same
initial state. Run Euler for 500 steps through time 5, and Verlet for 5000
steps through time 50. Use the first 500 Verlet steps for the short comparison;
the longer series is the continuation of that same Verlet trajectory.

Compute total energy as
`E(t) = 0.5 * (|velocity_i|^2 + |velocity_j|^2) + U(r(t))`.
Count the pair potential once. The reference is `E0 = U(1.2)` for both methods.
Report signed energy error `E(t) - E0`, without taking its absolute value or
normalizing it. Both initial errors are zero.

## Outputs

Replace the field-plot executable in `week2/md/src/main.rs` with this experiment.
Use the existing Plotters dependency and standard-library CSV writing; add no
dependencies. Running `cargo run --manifest-path week2/md/Cargo.toml` produces:

- `week2/energy_error.csv`, with header
  `method,step,time,total_energy,energy_error`.
  Write 501 `euler` rows followed by 5001 `verlet` rows, ordered by step within
  each method, including step zero for both. Preserve normal `f64` output
  precision rather than rounding errors to a fixed number of decimals.
- `week2/energy_error.png`, with two panels: Euler and Verlet error through
  time 5, and Verlet error through time 50. Give each panel its own vertical
  scale so the long-run Verlet behavior is visible. Label time, signed energy
  error, and integrator identity clearly.

Print output paths after successful generation. Propagate file and plotting
errors. Do not clamp forces, soften the potential, replace nonfinite data, or
silently omit failed samples; invalid simulation results must fail explicitly.

Remove the superseded tracked `week2/field.png` and exclude the generated CSV
and PNG from Git. Keep other unrelated repository content unchanged.

## Verification

Use focused Rust tests to verify the following semantics:

- Forces are equal and opposite, point toward the other atom at separation
  1.2, and use both displacement coordinates for a non-axis-aligned pair.
- Total energy includes both kinetic terms and one pair potential.
- A single step of each integrator agrees with its stated update formulas;
  the Euler check uses nonzero initial velocity to distinguish update order.
- The shared runner includes step zero, produces the requested number of
  samples, and computes the final time correctly.
- The specified trajectories remain finite; momentum stays near its initial
  zero value and the prescribed trajectory retains zero y coordinates.
- Verlet has smaller maximum absolute energy error than Euler over the common
  first 500 steps, and its maximum absolute error stays below `0.01 * |E0|`
  over the complete 5000-step run. This one-percent bound is an acceptance
  criterion, not a claim about the measured error; do not snapshot the trajectory.

Run Cargo tests and formatting checks. Generate and visually inspect the plot,
and verify the CSV header, row counts, final times, and initial zero errors.
Generated outputs are verification artifacts, not committed source files.

## Scope

This change implements only the two-atom experiment. No general particle
framework, force-model trait, integrator registry, runtime dispatch layer,
configuration system, thermostat, or boundary-condition abstraction is needed.
The requested integrator trait is the only new shared interface.
