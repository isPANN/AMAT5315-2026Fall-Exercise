# Two-atom Molecular Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare forward Euler and velocity-Verlet for the specified two-atom experiment, producing energy-error CSV and a two-panel plot.

**Architecture:** Keep the two-atom physics, one integrator trait, its two implementations, and one shared runner in the existing library. Replace the executable's field rendering with experiment execution and output generation.

**Tech Stack:** Rust 2024, standard library, existing Plotters 0.3.7 dependency and enabled features.

**Spec:** `week2/docs/superpowers/specs/2026-09-08-two-atom-md-design.md`

## Global Constraints

- Two atoms in two dimensions, each with mass 1.
- Reduced Lennard–Jones units: epsilon and sigma are both 1.
- Open boundaries; no walls, periodic wrapping, cutoff, or potential shifting.
- Initial positions: atom i at `(0.0, 0.0)`, atom j at `(1.2, 0.0)`.
- Both initial velocities are `(0.0, 0.0)`.
- Use a fixed time step of `0.01` for both methods, starting each from the same initial state.
- Run Euler for 500 steps through time 5, and Verlet for 5000 steps through time 50.
- The reference is `E0 = U(1.2)` for both methods.
- Report signed energy error `E(t) - E0`, without taking its absolute value or normalizing it.
- Use the existing Plotters dependency and standard-library CSV writing; add no dependencies.
- The requested integrator trait is the only new shared interface.
- Generated outputs are verification artifacts, not committed source files.

## File map and execution notes

- Modify `week2/md/src/lib.rs`: add state, forces, energy, integrators, runner, and focused unit tests. Preserve existing scalar Lennard–Jones functions and their tests. Leave the unrelated greeting function and test alone.
- Replace `week2/md/src/main.rs`: run the experiment, write CSV, render both plot panels.
- Create `week2/.gitignore`: ignore the two new generated outputs.
- Delete `week2/field.png`: superseded field-plot artifact.
- Keep `Cargo.toml`, `Cargo.lock`, and `week2/md/.gitignore` unchanged.

Commands below run from the repository root. Read the spec and applicable repository instructions before execution. Read the installed test-driven-development skill before implementation. The execution skills named in the header were not among the four installed skills; resolve their availability before invoking them, and do not silently install additional skills. This plan does not authorize extra abstractions, extra dependencies, or a PR.

---

## Task 1: Two-dimensional physics and both integrators

**Files:** Modify and test `week2/md/src/lib.rs`.

**Interfaces:**
- Consumes: existing `lennard_jones_energy(r: f64) -> f64` and `lennard_jones_force(r: f64) -> f64`.
- Produces: `State`, `State::forces() -> [[f64; 2]; 2]`, `State::energy() -> f64`, `Integrator::step(&self, &mut State, f64)`, `ForwardEuler`, and `VelocityVerlet`.

- [x] **Step 1: Add the following tests to the existing test module.** Add `use super::*;` and remove the now-redundant `use super::greeting;` import. Keep the existing tests.

```rust
#[test]
fn pair_force_and_total_energy() {
    let state = State {
        positions: [[0.0, 0.0], [0.72, 0.96]],
        velocities: [[1.0, 2.0], [-3.0, 4.0]],
    };
    let force = state.forces();
    let radial = lennard_jones_force(1.2);
    assert!((force[1][0] - 0.6 * radial).abs() < 1e-12);
    assert!((force[1][1] - 0.8 * radial).abs() < 1e-12);
    assert!(force[0][0] > 0.0 && force[0][1] > 0.0);
    for axis in 0..2 {
        assert_eq!(force[0][axis], -force[1][axis]);
    }
    assert!((state.energy() - (15.0 + lennard_jones_energy(1.2))).abs() < 1e-12);
}

#[test]
fn integrator_steps_follow_their_formulas() {
    let initial = State {
        positions: [[0.0, 0.0], [1.2, 0.0]],
        velocities: [[0.1, 0.2], [-0.1, -0.2]],
    };
    let dt = 0.01;
    let acceleration = initial.forces();
    let mut euler = initial;
    ForwardEuler.step(&mut euler, dt);
    let mut verlet = initial;
    VelocityVerlet.step(&mut verlet, dt);
    let mut moved = initial;
    for atom in 0..2 {
        for axis in 0..2 {
            let x = initial.positions[atom][axis];
            let v = initial.velocities[atom][axis];
            let a = acceleration[atom][axis];
            assert!((euler.positions[atom][axis] - (x + dt * v)).abs() < 1e-12);
            assert!((euler.velocities[atom][axis] - (v + dt * a)).abs() < 1e-12);
            moved.positions[atom][axis] = x + dt * v + 0.5 * dt * dt * a;
            assert!((verlet.positions[atom][axis] - moved.positions[atom][axis]).abs() < 1e-12);
        }
    }
    let new_acceleration = moved.forces();
    for atom in 0..2 {
        for axis in 0..2 {
            let expected = initial.velocities[atom][axis]
                + 0.5 * dt * (acceleration[atom][axis] + new_acceleration[atom][axis]);
            assert!((verlet.velocities[atom][axis] - expected).abs() < 1e-12);
        }
    }
}
```

- [x] **Step 2: Run the tests and confirm the missing types cause failure.**

```sh
cargo test --manifest-path week2/md/Cargo.toml
```

- [x] **Step 3: Add the direct implementation below the scalar Lennard–Jones functions.**

```rust
#[derive(Clone, Copy, Debug)]
pub struct State {
    pub positions: [[f64; 2]; 2],
    pub velocities: [[f64; 2]; 2],
}

impl State {
    pub fn forces(&self) -> [[f64; 2]; 2] {
        let dx = self.positions[1][0] - self.positions[0][0];
        let dy = self.positions[1][1] - self.positions[0][1];
        let r = dx.hypot(dy);
        let scale = lennard_jones_force(r) / r;
        [[-scale * dx, -scale * dy], [scale * dx, scale * dy]]
    }

    pub fn energy(&self) -> f64 {
        let dx = self.positions[1][0] - self.positions[0][0];
        let dy = self.positions[1][1] - self.positions[0][1];
        0.5 * self.velocities.iter().flatten().map(|v| v * v).sum::<f64>()
            + lennard_jones_energy(dx.hypot(dy))
    }
}

pub trait Integrator {
    fn step(&self, state: &mut State, dt: f64);
}

pub struct ForwardEuler;
pub struct VelocityVerlet;

impl Integrator for ForwardEuler {
    fn step(&self, state: &mut State, dt: f64) {
        let acceleration = state.forces();
        for (atom, force) in acceleration.iter().enumerate() {
            for (axis, &a) in force.iter().enumerate() {
                state.positions[atom][axis] += dt * state.velocities[atom][axis];
                state.velocities[atom][axis] += dt * a;
            }
        }
    }
}

impl Integrator for VelocityVerlet {
    fn step(&self, state: &mut State, dt: f64) {
        let old = state.forces();
        for (atom, force) in old.iter().enumerate() {
            for (axis, &a) in force.iter().enumerate() {
                state.positions[atom][axis] += dt * state.velocities[atom][axis]
                    + 0.5 * dt * dt * a;
            }
        }
        let new = state.forces();
        for atom in 0..2 {
            for axis in 0..2 {
                state.velocities[atom][axis] += 0.5 * dt * (old[atom][axis] + new[atom][axis]);
            }
        }
    }
}
```

- [x] **Step 4: Format, rerun the tests, and commit only the library.**

```sh
cargo fmt --manifest-path week2/md/Cargo.toml
cargo test --manifest-path week2/md/Cargo.toml
git add -- week2/md/src/lib.rs
git commit -m "Add two-atom state and Euler and Verlet integrators"
```

## Task 2: Shared experiment runner and trajectory checks

**Files:** Modify and test `week2/md/src/lib.rs`.

**Interfaces:**
- Consumes: `State`, `Integrator`, and both concrete integrators from Task 1.
- Produces: `INITIAL_STATE: State`, `Sample { step: usize, time: f64, total_energy: f64, energy_error: f64 }`, and `run(integrator: &impl Integrator, state: State, dt: f64, steps: usize) -> Vec<Sample>`.
- `run` computes the baseline from its supplied initial state's energy; for the required `INITIAL_STATE` that equals `U(1.2)` exactly. This keeps the runner consistent with its initial-state argument.

- [x] **Step 1: Add the following test.** The inner check exercises each concrete method through the same trait without adding production dispatch machinery.

```rust
#[test]
fn experiment_sampling_and_energy_behavior() {
    fn check(method: &impl Integrator, steps: usize) -> Vec<Sample> {
        let samples = run(method, INITIAL_STATE, 0.01, steps);
        assert_eq!(samples.len(), steps + 1);
        assert_eq!(samples[0].step, 0);
        assert_eq!(samples[0].time, 0.0);
        assert_eq!(samples[0].total_energy, lennard_jones_energy(1.2));
        assert_eq!(samples[0].energy_error, 0.0);
        assert_eq!(samples[steps].step, steps);
        assert_eq!(samples[steps].time, steps as f64 * 0.01);
        let mut state = INITIAL_STATE;
        for (step, sample) in samples.iter().enumerate() {
            if step > 0 {
                method.step(&mut state, 0.01);
            }
            assert_eq!(sample.step, step);
            assert_eq!(sample.time, step as f64 * 0.01);
            assert_eq!(sample.total_energy, state.energy());
            assert_eq!(sample.energy_error, state.energy() - lennard_jones_energy(1.2));
            assert!(sample.total_energy.is_finite());
            for value in state.positions.iter().chain(&state.velocities).flatten() {
                assert!(value.is_finite());
            }
            for axis in 0..2 {
                assert!((state.velocities[0][axis] + state.velocities[1][axis]).abs() < 1e-12);
            }
            for atom in 0..2 {
                assert_eq!(state.positions[atom][1], 0.0);
                assert_eq!(state.velocities[atom][1], 0.0);
            }
        }
        samples
    }
    let euler = check(&ForwardEuler, 500);
    let verlet = check(&VelocityVerlet, 5000);
    let max_error = |samples: &[Sample]| {
        samples.iter().map(|s| s.energy_error.abs()).fold(0.0_f64, f64::max)
    };
    assert!(max_error(&verlet[..=500]) < max_error(&euler));
    // Acceptance threshold: one percent of the initial energy magnitude.
    assert!(max_error(&verlet) < 0.01 * lennard_jones_energy(1.2).abs());
}
```

- [x] **Step 2: Run the new test and confirm failure from missing runner symbols.**

```sh
cargo test --manifest-path week2/md/Cargo.toml experiment_sampling_and_energy_behavior
```

- [x] **Step 3: Add the runner and its data types.** A nonfinite energy fails at sample production instead of reaching CSV or plotting; no force alteration or recovery path is introduced.

```rust
pub const INITIAL_STATE: State = State {
    positions: [[0.0, 0.0], [1.2, 0.0]],
    velocities: [[0.0, 0.0]; 2],
};

pub struct Sample {
    pub step: usize,
    pub time: f64,
    pub total_energy: f64,
    pub energy_error: f64,
}

pub fn run(integrator: &impl Integrator, mut state: State, dt: f64, steps: usize) -> Vec<Sample> {
    let initial_energy = state.energy();
    let mut samples = Vec::with_capacity(steps + 1);
    for step in 0..=steps {
        if step > 0 {
            integrator.step(&mut state, dt);
        }
        let total_energy = state.energy();
        assert!(total_energy.is_finite(), "nonfinite total energy at step {step}");
        samples.push(Sample {
            step,
            time: step as f64 * dt,
            total_energy,
            energy_error: total_energy - initial_energy,
        });
    }
    samples
}
```

- [x] **Step 4: Format, run all tests, and commit.** Do not relax the acceptance bound to hide a failure; inspect force signs and stepping order first.

```sh
cargo fmt --manifest-path week2/md/Cargo.toml
cargo test --manifest-path week2/md/Cargo.toml
git add -- week2/md/src/lib.rs
git commit -m "Add shared molecular dynamics energy experiment"
```

## Task 3: CSV export and two-panel comparison plot

**Files:** Replace `week2/md/src/main.rs`, create `week2/.gitignore`, delete `week2/field.png`.

**Interfaces:**
- Consumes: `run`, `INITIAL_STATE`, `ForwardEuler`, `VelocityVerlet`, and the `Sample` fields from Task 2.
- Produces: `week2/energy_error.csv` and `week2/energy_error.png` when run through Cargo; no new public Rust interface.

- [x] **Step 1: Establish the output check before replacing the executable.** Run the current executable, then the Python check in Step 3. On the current source the check must fail because the required CSV is absent. This check is a one-time verification command, not a committed diagnostic script.

```sh
cargo run --manifest-path week2/md/Cargo.toml
```

- [x] **Step 2: Replace all of `main.rs` with the following.** Use `PathElement` for polylines because the existing Plotters feature selection does not enable `LineSeries`. Each panel derives its own range from its data, including zero; the prescribed trajectories have nonzero error ranges, so no speculative constant-series fallback is needed.

```rust
use md::{run, ForwardEuler, VelocityVerlet, INITIAL_STATE};
use plotters::prelude::*;
use std::{error::Error, fs::File, io::{BufWriter, Write}, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let euler = run(&ForwardEuler, INITIAL_STATE, 0.01, 500);
    let verlet = run(&VelocityVerlet, INITIAL_STATE, 0.01, 5000);
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let csv_path = directory.join("energy_error.csv");
    let png_path = directory.join("energy_error.png");
    let mut csv = BufWriter::new(File::create(&csv_path)?);
    writeln!(csv, "method,step,time,total_energy,energy_error")?;
    for (method, samples) in [("euler", &euler), ("verlet", &verlet)] {
        for sample in samples {
            writeln!(csv, "{},{},{},{},{}", method, sample.step, sample.time,
                sample.total_energy, sample.energy_error)?;
        }
    }
    csv.flush()?;

    let root = BitMapBackend::new(&png_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, 1));
    for (panel_index, panel) in panels.iter().enumerate() {
        let series = if panel_index == 0 {
            vec![("Forward Euler", euler.as_slice(), RED),
                 ("Velocity-Verlet", &verlet[..=500], BLUE)]
        } else {
            vec![("Velocity-Verlet", verlet.as_slice(), BLUE)]
        };
        let mut low = 0.0_f64;
        let mut high = 0.0_f64;
        for (_, samples, _) in &series {
            for sample in *samples {
                low = low.min(sample.energy_error);
                high = high.max(sample.energy_error);
            }
        }
        let padding = 0.08 * (high - low);
        let (title, end_time) = if panel_index == 0 {
            ("Euler and velocity-Verlet: first 500 steps", 5.0)
        } else {
            ("Velocity-Verlet: 5000 steps", 50.0)
        };
        let mut chart = ChartBuilder::on(panel)
            .caption(title, ("sans-serif", 24))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(100)
            .build_cartesian_2d(0.0..end_time, (low - padding)..(high + padding))?;
        chart.configure_mesh()
            .x_desc("Time (reduced units)")
            .y_desc("E(t) - E0 (reduced energy)")
            .y_label_formatter(&|value| format!("{value:.2e}"))
            .draw()?;
        for (label, samples, color) in series {
            let points: Vec<_> = samples.iter()
                .map(|sample| (sample.time, sample.energy_error)).collect();
            chart.draw_series(std::iter::once(PathElement::new(points, color.stroke_width(2))))?
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }
        chart.configure_series_labels()
            .background_style(WHITE.mix(0.9))
            .border_style(BLACK)
            .draw()?;
    }
    root.present()?;
    println!("{}\n{}", csv_path.display(), png_path.display());
    Ok(())
}
```

Create `week2/.gitignore` with exactly:

```gitignore
/energy_error.csv
/energy_error.png
```

Remove the superseded tracked artifact:

```sh
git rm -- week2/field.png
```

- [x] **Step 3: Run formatting, tests, the executable, and the CSV check.** The CSV check validates the public output contract and reports measured errors; it does not snapshot results.

```sh
cargo fmt --manifest-path week2/md/Cargo.toml
cargo fmt --manifest-path week2/md/Cargo.toml -- --check
cargo test --manifest-path week2/md/Cargo.toml
cargo run --manifest-path week2/md/Cargo.toml
python3 - <<'PY'
import csv
import math
from pathlib import Path

with Path('week2/energy_error.csv').open(newline='') as handle:
    reader = csv.DictReader(handle)
    assert reader.fieldnames == ['method', 'step', 'time', 'total_energy', 'energy_error']
    rows = list(reader)
assert len(rows) == 5502
assert [row['method'] for row in rows] == ['euler'] * 501 + ['verlet'] * 5001
reference = 4 * (1.2 ** -12 - 1.2 ** -6)
for method, count, end_time in [('euler', 501, 5.0), ('verlet', 5001, 50.0)]:
    samples = [row for row in rows if row['method'] == method]
    assert len(samples) == count
    assert float(samples[0]['energy_error']) == 0.0
    assert math.isclose(float(samples[0]['total_energy']), reference, abs_tol=1e-14)
    assert float(samples[-1]['time']) == end_time
    for step, row in enumerate(samples):
        assert int(row['step']) == step
        assert float(row['time']) == step * 0.01
        energy = float(row['total_energy'])
        error = float(row['energy_error'])
        assert math.isfinite(energy) and math.isfinite(error)
        assert math.isclose(error, energy - reference, abs_tol=1e-14)
    print(method, 'maximum absolute energy error:', max(abs(float(r['energy_error'])) for r in samples))
assert Path('week2/energy_error.png').is_file()
PY
```

- [x] **Step 4: Open `week2/energy_error.png` with the image-viewing tool.** Confirm both panels, correct time ranges, legible axis labels and legends, visible long-run Verlet oscillations, and no clipping or legend overlap obscuring the curves. Adjust only plotting layout if needed, rerun the executable, and inspect again after any layout change. Do not infer visual correctness from file existence.

- [x] **Step 5: Check scope and commit only source changes.** Confirm the CSV and PNG are ignored, the old field image is deleted, and there are no dependency changes. Report measured error values and checks in the completion message. If code review is requested, use the installed requesting-code-review skill and resolve the review before declaring completion.

```sh
git check-ignore week2/energy_error.csv week2/energy_error.png
git diff --check
git add -- week2/md/src/main.rs week2/.gitignore
git diff --cached --stat
git commit -m "Export molecular dynamics energy data and comparison plot"
git status --short
```

## Plan self-review

The three tasks cover the physical model, update formulas, shared runner, all run lengths and measurements, both output formats, replacement cleanup, and the specified verification. Every production symbol used by the executable is introduced in Tasks 1 or 2. No additional dependency, dispatch interface, generated artifact, or unrelated cleanup is included. Implementation is complete. All seven Rust tests, formatting, and CSV contract checks passed. The generated two-panel plot was visually inspected. Maximum absolute energy errors were 1.7423937126005058 for Euler through time 5 and 0.00028960755168305763 for Verlet through time 50.
