# Lennard–Jones Fluid CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver `md run`, `md check`, and `md video` with the approved contract run, saved-state analysis, and an MP4 below 2 MB.

**Architecture:** Evolve the existing state and integrators into one variable-particle core that supports the specified periodic fluid and the existing open dimer. Typed trajectory I/O is shared by check and video; analysis never advances simulation. Render with Plotters and encode with ffmpeg.

**Tech Stack:** Rust 2024; existing Plotters 0.3.7; Clap 4.5.48 with derive, Serde 1.0.228 with derive, serde_json 1.0.145 with float_roundtrip, rand 0.8.5, rand_chacha 0.3.1, rand_distr 0.4.3; ffmpeg with libx264, ffprobe; Make.

**Spec:** `week2/docs/superpowers/specs/2026-09-08-lennard-jones-fluid-cli-design.md`

## Global Constraints

- The executable is named `md`.
- Use two dimensions and reduced units with atom mass, epsilon, sigma, and Boltzmann constant all equal to 1.
- Require both box sides to exceed `2 * rc`, where `rc = 2.5`.
- Inside the cutoff the force remains the ordinary Lennard–Jones force; this is a potential shift, not a force shift.
- No tail correction, force softening, or force clamping is used.
- The fluid CLI exposes this integrator only.
- Do not save production step zero or an additional unscheduled final frame.
- The trajectory is the sole state source for check and video.
- Do not add a fourth physical acceptance gate comparing stored and recomputed energy.
- Use the specified rho normalization, without a finite-N correction.
- Do not drop frames or truncate the trajectory to meet the size limit.
- Exclude generated trajectories, videos, rendered frames, and build artifacts from commits.

Numerical constants: n=100, rho=0.8, temperature=0.5, dt=0.01, eq_steps=2000, steps=10000, sample_every=50, seed=2026; initial and every-50-equilibration-step rescaling; drift <0.002, temperature deviation <0.05, reduced speed chi-squared <2; 24 speed bins, denominator 22; 100 RDF bins; 30 fps; MP4 <2,000,000 bytes.

## Files and execution

| File | Responsibility |
| --- | --- |
| `week2/md/src/lib.rs` | Shared physical state, pair interactions, integrators, existing dimer energy experiment |
| `week2/md/src/trajectory.rs` | Run metadata, initialization, production writing, saved-file validation |
| `week2/md/src/analysis.rs` | Recomputed physical checks, cumulative RDF, contrast |
| `week2/md/src/video.rs` | Plotters frames and ffmpeg encoding |
| `week2/md/src/main.rs` | Clap CLI and exit status |
| `week2/md/tests/cli.rs` | Process-level output and error checks |
| `week2/md/Cargo.toml`, `Cargo.lock` | Dependencies and resolved versions |
| `week2/md/examples/dimer.rs` | Update calls to the shared core; preserve its plot |
| `week2/Makefile`, `week2/README.md`, `week2/.gitignore` | Reproduction commands, usage, generated-artifact exclusions |

All commands below run from the repository root unless `make -C week2` is used. Before implementation read the installed TDD skill. The execution helpers named in the header were not installed with the four requested skills; use the user's authorized inline workflow if they remain absent, without installing extra skills or requiring another permission cycle. Do not dispatch agents unless the user chooses delegation.

The untracked README and dimer example are user work. Read them before editing and retain their content; stage only their necessary integration changes. Never stage `week2/dimer.png`. No PR is requested. If a PR is later requested and exceeds the repository's 20-file or 1,000-added-line limit, confirm that PR scope before creating it.

Dependency versions above are Cargo version requirements, not implementation freezes; let Cargo resolve and record its normal lockfile. The algorithm choice is ChaCha8 seeded with `seed_from_u64`, and `Normal::new(0.0, temperature.sqrt())`. Do not change the generator to hunt for a passing contract result.

Reference documentation: [Clap derive](https://docs.rs/clap/4.5.48/clap/), [typed Serde JSON](https://docs.rs/serde_json/1.0.145/serde_json/), [Gaussian distributions](https://docs.rs/rand_distr/0.4.3/rand_distr/), [ChaCha generator](https://docs.rs/rand_chacha/latest/rand_chacha/), [ffmpeg encoding options](https://ffmpeg.org/ffmpeg.html). Use the selected version's locally downloaded Rust source when checking exact APIs; do not assume the latest RNG API matches 0.3.1.

---

## Task 1: Evolve the shared physical core

**Files:** `week2/md/src/lib.rs`, `week2/md/examples/dimer.rs`, and the current `week2/md/src/main.rs` call sites until Task 5 replaces the binary.

**Interfaces:** Introduce `pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>`. Replace fixed arrays with `State { positions: Vec<[f64;2]>, velocities: Vec<[f64;2]>, box_size: Option<[f64;2]> }`, deriving Clone and Debug. `None` is the existing open, unshifted dimer; `Some(box)` is the specified periodic, shifted fluid. This field represents the two required physical cases, not a general force-model interface.

Expose `RC: f64 = 2.5`, `State::displacement(i,j)->[f64;2]`, `State::interactions()->Result<(Vec<[f64;2]>, f64)>`, `State::kinetic_energy()->f64`, `State::energy()->Result<f64>`, and `Integrator::step(&self, &mut State, dt:f64)->Result<()>`. Replace `INITIAL_STATE` with `dimer_state()->State`. Keep `ForwardEuler`, `VelocityVerlet`, and the dimer `Sample` fields. Change `run(&impl Integrator, State, f64, usize)->Result<Vec<Sample>>` to propagate errors. Remove the old fixed-array implementation and obsolete `forces` method after updating callers.

- [x] **1. Add a failing periodic-interaction test.** The literals catch wrong minimum-image direction, a force shift, double-counted potential, and missing cutoff.

```rust
#[test]
fn periodic_pair_uses_shifted_energy_and_plain_force() {
    let mut state = State {
        positions: vec![[0.0, 0.0], [5.0, 0.0]],
        velocities: vec![[0.0, 0.0]; 2],
        box_size: Some([6.0, 6.0]),
    };
    assert_eq!(state.displacement(0, 1), [-1.0, 0.0]);
    let (force, potential) = state.interactions().unwrap();
    assert_eq!(force, vec![[24.0, 0.0], [-24.0, 0.0]]);
    assert!((potential - 0.016316891136).abs() < 1e-12);
    state.positions[1] = [2.5, 0.0];
    let (force, potential) = state.interactions().unwrap();
    assert_eq!(force, vec![[0.0, 0.0]; 2]);
    assert_eq!(potential, 0.0);
    state.positions[1] = [0.0, 0.0];
    assert!(state.interactions().is_err());
}
```

Run `cargo test --manifest-path week2/md/Cargo.toml periodic_pair` and observe failure from missing periodic state/API.

- [x] **2. Implement the interaction calculation.** Keep the existing scalar functions, use a direct unordered pair loop, and calculate kinetic energy as the existing component-square sum.

```rust
pub fn displacement(&self, i: usize, j: usize) -> [f64; 2] {
    let mut d = std::array::from_fn(|axis| self.positions[j][axis] - self.positions[i][axis]);
    if let Some(lengths) = self.box_size {
        for axis in 0..2 { d[axis] -= lengths[axis] * (d[axis] / lengths[axis]).round(); }
    }
    d
}

pub fn interactions(&self) -> Result<(Vec<[f64; 2]>, f64)> {
    let mut forces = vec![[0.0; 2]; self.positions.len()];
    let mut potential = 0.0;
    // ponytail: O(N²) pair loop; use cell lists if larger-run timings require them.
    for i in 0..self.positions.len() {
        for j in i + 1..self.positions.len() {
            let d = self.displacement(i, j);
            let r2 = d[0]*d[0] + d[1]*d[1];
            if !r2.is_finite() || r2 <= 0.0 { return Err(format!("invalid separation for atoms {i}, {j}").into()); }
            if self.box_size.is_some() && r2 >= RC*RC { continue; }
            let r = r2.sqrt();
            let u = lennard_jones_energy(r);
            let scale = lennard_jones_force(r) / r;
            if !u.is_finite() || !scale.is_finite() { return Err(format!("nonfinite interaction for atoms {i}, {j}").into()); }
            potential += u - if self.box_size.is_some() { lennard_jones_energy(RC) } else { 0.0 };
            for axis in 0..2 {
                forces[i][axis] -= scale*d[axis];
                forces[j][axis] += scale*d[axis];
            }
        }
    }
    Ok((forces, potential))
}

pub fn kinetic_energy(&self) -> f64 {
    0.5 * self.velocities.iter().flatten().map(|v| v*v).sum::<f64>()
}

pub fn energy(&self) -> Result<f64> {
    let energy = self.interactions()?.1 + self.kinetic_energy();
    if !energy.is_finite() { return Err("nonfinite total energy".into()); }
    Ok(energy)
}
```

Do not add internal repeated shape validation: constructors/file readers own vector-shape validation. Boundary invalidity and singular forces must fail explicitly.

- [x] **3. Update integration and all dimer consumers together.** Old one-step tests already cover the formulas: convert their state literals to vectors with `box_size: None`, clone states instead of copying, and unwrap successful Results. Retain the independent numeric expectations. Replace the former panic expectation with `assert!(run(...).is_err())`.

Velocity-Verlet method body:

```rust
let old = state.interactions()?.0;
for (atom, acceleration) in old.iter().enumerate() {
    for axis in 0..2 {
        state.positions[atom][axis] += dt*state.velocities[atom][axis] + 0.5*dt*dt*acceleration[axis];
        if let Some(lengths) = state.box_size {
            state.positions[atom][axis] = state.positions[atom][axis].rem_euclid(lengths[axis]);
        }
    }
}
let new = state.interactions()?.0;
for atom in 0..state.positions.len() {
    for axis in 0..2 { state.velocities[atom][axis] += 0.5*dt*(old[atom][axis] + new[atom][axis]); }
}
Ok(())
```

Euler uses the same old-force loop with `x += dt*v` before `v += dt*a`, and the same periodic wrapping. `run` uses `integrator.step(...)?` and `state.energy()?`, returning `Ok(samples)`. Construct the dimer directly:

```rust
pub fn dimer_state() -> State {
    State { positions: vec![[0.0, 0.0], [1.2, 0.0]], velocities: vec![[0.0, 0.0]; 2], box_size: None }
}
```

Update the example and interim binary to `run(&ForwardEuler, dimer_state(), 0.01, 500)?`, the analogous Verlet call, and `dimer_state().energy()?.abs()`. Preserve the example's plot labels and normalization. No duplicate dimer engine.

- [x] **4. Run checks and commit.**

```sh
cargo fmt --manifest-path week2/md/Cargo.toml
cargo test --manifest-path week2/md/Cargo.toml
cargo check --manifest-path week2/md/Cargo.toml --examples
git add -- week2/md/src/lib.rs week2/md/src/main.rs week2/md/examples/dimer.rs
git diff --cached --check
git commit -m "Extend shared molecular dynamics core to periodic fluids"
```

## Task 2: Initialize, simulate, and read the trajectory contract

**Files:** Create `week2/md/src/trajectory.rs`; add `pub mod trajectory;` to lib.rs; update Cargo.toml/Cargo.lock. Import `crate::{State,RC,Result,Integrator,VelocityVerlet}` and the standard-library I/O/path types named below.

**Interfaces:** All definitions below are public within `md::trajectory`: `RunConfig`, `Frame`, `geometry(n:usize,rho:f64)->Result<[f64;2]>`, `rescale(&mut State,target:f64)->Result<()>`, `initial_state(&RunConfig)->Result<State>`, `write_run(&RunConfig,&Path)->Result<()>`, `read_run(&Path)->Result<(RunConfig,Vec<Frame>)>`. `Frame::state(&self,&RunConfig)->State` constructs a cloned saved physical state without stepping. `RunConfig::validate()->Result<()>` and `Frame::validate(&self,&RunConfig,index:usize)->Result<()>` enforce the file contract; index is zero-based.

- [x] **1. Add dependencies and the initial failing test.** Keep existing Plotters settings unchanged.

```toml
clap = { version = "4.5.48", features = ["derive"] }
serde = { version = "1.0.228", features = ["derive"] }
serde_json = { version = "1.0.145", features = ["float_roundtrip"] }
rand = "0.8.5"
rand_chacha = "0.3.1"
rand_distr = "0.4.3"
```

Place this test inside trajectory.rs with `use super::*;`. First provide the config data structure in Step 2 so the missing `initial_state` identifies the missing behavior.

```rust
#[test]
fn seeded_lattice_has_zero_momentum_and_target_kinetic_energy() {
    let config = RunConfig::default();
    let a = initial_state(&config).unwrap();
    let b = initial_state(&config).unwrap();
    assert_eq!(a.positions, b.positions);
    assert_eq!(a.velocities, b.velocities);
    assert_eq!(a.positions.len(), 100);
    assert!((a.kinetic_energy() - 49.5).abs() < 1e-11);
    for axis in 0..2 {
        assert!(a.velocities.iter().map(|v| v[axis]).sum::<f64>().abs() < 1e-12);
    }
    let lengths = a.box_size.unwrap();
    assert!((lengths[0]*lengths[1] - 125.0).abs() < 1e-10);
    assert!((a.positions[10][0] - a.positions[1][0]/2.0).abs() < 1e-12);
    for n in [100, 400, 1600] {
        let lengths = geometry(n, 0.8).unwrap();
        assert!((lengths[0]*lengths[1] - n as f64/0.8).abs() < 1e-9);
    }
    assert!(geometry(99, 0.8).is_err());
    assert!(geometry(81, 0.8).is_err());
}
```

Run `cargo test --manifest-path week2/md/Cargo.toml seeded_lattice` before implementing initialization.

- [x] **2. Implement config, geometry, and thermostat.** Derive typed serialization with exactly the specified JSON field names.

```rust
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RunConfig {
    pub n: usize, pub rho: f64,
    #[serde(rename = "box")] pub box_size: [f64; 2],
    pub dt: f64, pub temperature: f64, pub eq_steps: usize,
    pub steps: usize, pub sample_every: usize, pub seed: u64,
    pub integrator: String,
}
impl Default for RunConfig {
    fn default() -> Self {
        Self { n:100, rho:0.8, box_size:geometry(100,0.8).unwrap(), dt:0.01,
            temperature:0.5, eq_steps:2000, steps:10000, sample_every:50,
            seed:2026, integrator:"velocity-verlet".into() }
    }
}
pub fn geometry(n: usize, rho: f64) -> Result<[f64; 2]> {
    let q = n.isqrt();
    if q < 2 || q*q != n || q%2 != 0 { return Err("n must be an even square grid".into()); }
    if !rho.is_finite() || rho <= 0.0 { return Err("rho must be finite and positive".into()); }
    let a = (2.0/(3.0_f64.sqrt()*rho)).sqrt();
    let lengths = [q as f64*a, q as f64*a*3.0_f64.sqrt()/2.0];
    if lengths.iter().any(|v| !v.is_finite() || *v <= 2.0*RC) { return Err("box sides must exceed 2*rc".into()); }
    Ok(lengths)
}
pub fn rescale(state: &mut State, target: f64) -> Result<()> {
    let t = 2.0*state.kinetic_energy()/(2*state.velocities.len()-2) as f64;
    if !t.is_finite() || t <= 0.0 { return Err("undefined thermostat temperature".into()); }
    let scale = (target/t).sqrt();
    for v in state.velocities.iter_mut().flatten() { *v *= scale; }
    Ok(())
}
```

Implement the metadata validation used by both initialization and the saved-file reader:

```rust
impl RunConfig {
    pub fn validate(&self)->Result<()> {
        let expected=geometry(self.n,self.rho)?;
        for axis in 0..2 {
            if !self.box_size[axis].is_finite() || (self.box_size[axis]-expected[axis]).abs()>1e-12*expected[axis].abs().max(1.0) {
                return Err("box does not match n and rho".into());
            }
        }
        for (field,value) in [("dt",self.dt),("temperature",self.temperature)] {
            if !value.is_finite() || value<=0.0 { return Err(format!("{field} must be finite and positive").into()); }
        }
        if self.sample_every==0 || self.steps<self.sample_every { return Err("run must save at least one production frame".into()); }
        if !(self.steps as f64*self.dt).is_finite() { return Err("nonfinite production duration".into()); }
        if self.integrator!="velocity-verlet" { return Err("unsupported integrator".into()); }
        Ok(())
    }
}
```

Initialization body (imports `rand::SeedableRng`, `rand_chacha::ChaCha8Rng`, `rand_distr::{Distribution, Normal}`):

```rust
config.validate()?;
let q = config.n.isqrt();
let a = config.box_size[0]/q as f64;
let h = config.box_size[1]/q as f64;
let positions = (0..q).flat_map(|j| (0..q).map(move |i| [(i as f64+0.5*(j%2) as f64)*a,j as f64*h])).collect();
let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
let normal = Normal::new(0.0, config.temperature.sqrt())?;
let mut velocities: Vec<[f64;2]> = (0..config.n).map(|_| [normal.sample(&mut rng),normal.sample(&mut rng)]).collect();
for axis in 0..2 {
    let mean = velocities.iter().map(|v| v[axis]).sum::<f64>()/config.n as f64;
    for v in &mut velocities { v[axis] -= mean; }
}
let mut state = State { positions, velocities, box_size:Some(config.box_size) };
rescale(&mut state,config.temperature)?;
Ok(state)
```

- [x] **3. Add a failing production-file test.** Use one test-owned directory under `std::env::temp_dir()` named with process ID and this semantic test name; create it with `create_dir`, never delete an unknown pre-existing directory. Cleanup only after assertions succeed.

```rust
#[test]
fn production_files_exclude_zero_and_validate_sampling() {
    let dir = std::env::temp_dir().join(format!("md-production-{}",std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let config = RunConfig { eq_steps:51, steps:61, sample_every:10, ..RunConfig::default() };
    write_run(&config,&dir).unwrap();
    let (saved, frames) = read_run(&dir).unwrap();
    assert_eq!(saved.eq_steps,51);
    assert_eq!(frames.iter().map(|f| f.step).collect::<Vec<_>>(),vec![10,20,30,40,50,60]);
    assert_eq!(frames[0].t,0.1);
    assert_eq!(frames[5].t,0.6);
    let mut manual = initial_state(&config).unwrap();
    for step in 1..=111 {
        VelocityVerlet.step(&mut manual,config.dt).unwrap();
        if step == 50 { rescale(&mut manual,config.temperature).unwrap(); }
    }
    assert_eq!(frames[5].pos,manual.positions);
    assert_eq!(frames[5].vel,manual.velocities);
    let mut bad = frames[0].clone();
    bad.step = 0;
    assert!(bad.validate(&config,0).is_err());
    bad = frames[0].clone(); bad.pos[0][0] = -0.1;
    assert!(bad.validate(&config,0).is_err());
    std::fs::write(dir.join("traj.jsonl"),"{bad json}\n").unwrap();
    assert!(read_run(&dir).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
```

Run `cargo test --manifest-path week2/md/Cargo.toml production_files` and observe failure before file I/O implementation.

- [x] **4. Implement frames and file I/O.**

```rust
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Frame {
    pub step:usize, pub t:f64, pub pos:Vec<[f64;2]>, pub vel:Vec<[f64;2]>,
    #[serde(rename="E_pot")] pub e_pot:f64,
    #[serde(rename="E_kin")] pub e_kin:f64,
}
impl Frame {
    pub fn state(&self, config:&RunConfig)->State {
        State { positions:self.pos.clone(), velocities:self.vel.clone(), box_size:Some(config.box_size) }
    }
}
```

Fixed `[f64;2]` deserialization enforces component count. The reader attaches the line number to these field-specific errors:

```rust
impl Frame {
    pub fn validate(&self,config:&RunConfig,index:usize)->Result<()> {
        if self.step!=(index+1)*config.sample_every { return Err("invalid production sampling sequence".into()); }
        let expected=self.step as f64*config.dt;
        if !self.t.is_finite() || (self.t-expected).abs()>1e-12*expected.abs().max(1.0) { return Err("t does not equal step*dt".into()); }
        if self.pos.len()!=config.n || self.vel.len()!=config.n { return Err("particle count mismatch".into()); }
        if self.pos.iter().chain(&self.vel).flatten().any(|v| !v.is_finite()) || !self.e_pot.is_finite() || !self.e_kin.is_finite() {
            return Err("nonfinite frame data".into());
        }
        for p in &self.pos {
            for axis in 0..2 {
                if p[axis]<0.0 || p[axis]>=config.box_size[axis] { return Err("position is outside the wrapped box".into()); }
            }
        }
        Ok(())
    }
}
```

Use this write body with `std::fs`, `std::io::{BufWriter,Write}`, and `std::path::Path`:

```rust
let mut state = initial_state(config)?;
std::fs::create_dir_all(out)?;
let mut metadata = BufWriter::new(std::fs::File::create(out.join("run.json"))?);
serde_json::to_writer_pretty(&mut metadata,config)?; metadata.flush()?;
let mut file = BufWriter::new(std::fs::File::create(out.join("traj.jsonl"))?);
for step in 1..=config.eq_steps {
    VelocityVerlet.step(&mut state,config.dt).map_err(|e| format!("equilibration step {step}: {e}"))?;
    if step%50 == 0 { rescale(&mut state,config.temperature)?; }
}
for step in 1..=config.steps {
    VelocityVerlet.step(&mut state,config.dt).map_err(|e| format!("production step {step}: {e}"))?;
    if step%config.sample_every == 0 {
        let frame = Frame { step,t:step as f64*config.dt,pos:state.positions.clone(),vel:state.velocities.clone(),
            e_pot:state.interactions()?.1,e_kin:state.kinetic_energy() };
        frame.validate(config,step/config.sample_every-1)?;
        serde_json::to_writer(&mut file,&frame)?; writeln!(file)?;
    }
}
file.flush()?;
Ok(())
```

Reader body, additionally importing `BufRead` and `BufReader`:

```rust
let config: RunConfig = serde_json::from_reader(BufReader::new(std::fs::File::open(dir.join("run.json"))?))?;
config.validate()?;
let mut frames = Vec::new();
for (index,line) in BufReader::new(std::fs::File::open(dir.join("traj.jsonl"))?).lines().enumerate() {
    let frame:Frame = serde_json::from_str(&line?).map_err(|e| format!("frame {}: {e}",index+1))?;
    frame.validate(&config,index).map_err(|e| format!("frame {}: {e}",index+1))?;
    frame.state(&config).energy().map_err(|e| format!("frame {}: {e}",index+1))?;
    frames.push(frame);
}
if frames.len() != config.steps/config.sample_every { return Err("trajectory frame count does not match run.json".into()); }
Ok((config,frames))
```

- [x] **5. Run all tests and commit only task files.**

```sh
cargo fmt --manifest-path week2/md/Cargo.toml
cargo test --manifest-path week2/md/Cargo.toml
git add -- week2/md/src/lib.rs week2/md/src/trajectory.rs week2/md/Cargo.toml week2/md/Cargo.lock
git diff --cached --check
git commit -m "Save and validate seeded fluid trajectories"
```

## Task 3: Saved-state physics checks and cumulative pair structure

**Files:** Create `week2/md/src/analysis.rs`; add `pub mod analysis;` to lib.rs. Import `crate::Result` and `crate::trajectory::{RunConfig,Frame}`.

**Interfaces:** `energy_drift(&[f64])->Result<f64>`, `speed_statistics(&[f64])->Result<(f64,f64)>` consumes actual speeds and returns (T_speed,reduced_chi_squared); `check(&RunConfig,&[Frame])->Result<Report>`; `Report { drift:f64, temperature:f64, speed_shape:f64 }`; `Report::passed(target:f64)->bool`; `rdf_series(&RunConfig,&[Frame])->Vec<Rdf>`; `Rdf { radius:Vec<f64>, values:Vec<f64>, contrast:f64 }` with all fields public. These functions consume validated saved input; no stepping or RNG imports.

- [x] **1. Add failing hand-calculated analysis tests.** Include the stored-energy and strict-limit tests specified in Step 3 in this initial failing-test run, before writing the analysis implementation.

```rust
#[test]
fn drift_averages_end_windows_and_speed_shape_uses_22_degrees() {
    let mut energies = vec![-10.0;20];
    energies[18] = -9.99; energies[19] = -9.99;
    assert!((energy_drift(&energies).unwrap()-0.001).abs() < 1e-12);
    let (temperature,shape) = speed_statistics(&[1.0;24]).unwrap();
    assert_eq!(temperature,0.5);
    // All 24 speeds occupy one bin: (23² + 23)/22.
    assert!((shape-552.0/22.0).abs() < 1e-12);
    assert!(energy_drift(&[0.0]).is_err());
    assert!(speed_statistics(&[0.0;24]).is_err());
}

#[test]
fn rdf_counts_both_neighbours_and_accumulates_frames() {
    // Synthetic pair isolates ring normalization; it is not a lattice initializer test.
    let config = RunConfig { n:2,rho:2.0/36.0,box_size:[6.0,6.0],..RunConfig::default() };
    let first = Frame { step:50,t:0.5,pos:vec![[0.0,0.0],[1.0,0.0]],vel:vec![[0.0,0.0];2],e_pot:0.0,e_kin:0.0 };
    let mut second = first.clone(); second.pos[1][0]=1.5;
    let curves = rdf_series(&config,&[first,second]);
    let expected = 2.0/(2.0*(2.0/36.0)*std::f64::consts::PI*(1.02_f64.powi(2)-0.99_f64.powi(2)));
    assert!((curves[0].values[33]-expected).abs() < 1e-10);
    assert!((curves[1].values[33]-expected/2.0).abs() < 1e-10);
    assert_eq!(curves[0].contrast,1.0); // No pairs beyond radius 2: g=0 there.
}
```

Run `cargo test --manifest-path week2/md/Cargo.toml --lib` before implementing the new functions.

- [x] **2. Implement the exact statistics and recomputation path.**

```rust
pub fn energy_drift(energies:&[f64])->Result<f64> {
    if energies.is_empty() || energies[0] == 0.0 { return Err("undefined reference energy".into()); }
    let k = (energies.len()/10).max(1);
    let first = energies[..k].iter().sum::<f64>()/k as f64;
    let last = energies[energies.len()-k..].iter().sum::<f64>()/k as f64;
    let drift = (last-first).abs()/energies[0].abs();
    if !drift.is_finite() { return Err("nonfinite energy drift".into()); }
    Ok(drift)
}
pub fn speed_statistics(speeds:&[f64])->Result<(f64,f64)> {
    let temperature = speeds.iter().map(|v| v*v).sum::<f64>()/(2.0*speeds.len() as f64);
    if !temperature.is_finite() || temperature <= 0.0 { return Err("undefined speed temperature".into()); }
    let mut edges = [f64::INFINITY;25];
    for (k,edge) in edges[..24].iter_mut().enumerate() {
        *edge = (-2.0*temperature*(1.0-k as f64/24.0).ln()).sqrt();
    }
    let mut counts = [0usize;24];
    for speed in speeds {
        let bin = edges[1..24].partition_point(|edge| speed >= edge);
        counts[bin] += 1;
    }
    let expected = speeds.len() as f64/24.0;
    let shape = counts.iter().map(|&n| (n as f64-expected).powi(2)/expected).sum::<f64>()/22.0;
    Ok((temperature,shape))
}
pub struct Report { pub drift:f64, pub temperature:f64, pub speed_shape:f64 }
impl Report {
    pub fn passed(&self,target:f64)->bool {
        self.drift < 0.002 && (self.temperature-target).abs() < 0.05 && self.speed_shape < 2.0
    }
}
pub fn check(config:&RunConfig,frames:&[Frame])->Result<Report> {
    let energies = frames.iter().map(|f| f.state(config).energy()).collect::<Result<Vec<_>>>()?;
    let speeds:Vec<f64> = frames.iter().flat_map(|f| f.vel.iter().map(|v| v[0].hypot(v[1]))).collect();
    let (temperature,speed_shape) = speed_statistics(&speeds)?;
    Ok(Report { drift:energy_drift(&energies)?,temperature,speed_shape })
}
```

RDF body uses the already-tested minimum-image calculation; it does not use the force cutoff:

```rust
let rmax = config.box_size[0].min(config.box_size[1])/2.0;
let width = rmax/100.0;
let radius:Vec<f64> = (0..100).map(|bin| (bin as f64+0.5)*width).collect();
let mut counts = [0u64;100];
let mut curves = Vec::with_capacity(frames.len());
for (index,frame) in frames.iter().enumerate() {
    let state = frame.state(config);
    for i in 0..config.n {
        for j in i+1..config.n {
            let d = state.displacement(i,j); let r=d[0].hypot(d[1]);
            if r < rmax { counts[(r/width).floor() as usize] += 2; }
        }
    }
    let values:Vec<f64> = (0..100).map(|bin| {
        let inner=bin as f64*width; let outer=(bin+1) as f64*width;
        counts[bin] as f64/((index+1) as f64*config.n as f64*config.rho*std::f64::consts::PI*(outer*outer-inner*inner))
    }).collect();
    let long:Vec<f64> = radius.iter().zip(&values).filter(|(r,_)| **r>2.0).map(|(_,g)| (g-1.0).powi(2)).collect();
    let contrast = (long.iter().sum::<f64>()/long.len() as f64).sqrt();
    curves.push(Rdf { radius:radius.clone(),values,contrast });
}
curves
```

The box validation ensures rmax>2.5, hence plotted bins above radius 2 exist. No empty-bin fallback is needed. Define the `Rdf` struct listed in Interfaces before this function.

- [x] **3. Confirm the saved-energy independence and strict-limit tests added in Step 1 now pass.** The synthetic frames below isolate recomputation: calculate a report, change finite stored `e_pot/e_kin` to unrelated values, and calculate again. Identical report quantities establish that those fields do not control the result. Task 5 separately exercises actual files through the CLI.

```rust
#[test]
fn physical_checks_ignore_stored_energy_fields() {
    let config=RunConfig::default();
    let state=initial_state(&config).unwrap();
    let frame=Frame { step:50,t:0.5,pos:state.positions,vel:state.velocities,e_pot:0.0,e_kin:0.0 };
    let mut frames=vec![frame.clone(),frame];
    let a=check(&config,&frames).unwrap();
    frames[0].e_pot=99999.0; frames[1].e_kin=123.0;
    let b=check(&config,&frames).unwrap();
    assert_eq!(a.drift,b.drift);
    assert_eq!(a.temperature,b.temperature);
    assert_eq!(a.speed_shape,b.speed_shape);
}
```

This test deliberately exercises analysis independently of schedule validation; Task 2 tests the saved-file boundary. Add `use crate::trajectory::initial_state;` in the test module. Add a strict-limit check using `Report { drift:0.002,temperature:0.5,speed_shape:1.0 }` and `assert!(!report.passed(0.5))`; likewise set speed_shape=2.0 with drift=0.0. For temperature, use a clearly outside value 0.56 to avoid a floating-point tie at 0.55.

- [x] **4. Format, run tests, and commit.**

```sh
cargo fmt --manifest-path week2/md/Cargo.toml
cargo test --manifest-path week2/md/Cargo.toml
git add -- week2/md/src/lib.rs week2/md/src/analysis.rs
git diff --cached --check
git commit -m "Recompute trajectory physics and radial structure"
```

## Task 4: Render and encode the saved trajectory

**Files:** Create `week2/md/src/video.rs`; add `pub mod video;` to lib.rs. No additional Rust dependency.

**Interfaces:** `render_frame(config:&RunConfig,frame:&Frame,rdf:&Rdf,ymax:f64,path:&Path)->Result<()>`; `record(config:&RunConfig,frames:&[Frame],out:&Path)->Result<()>`. Use only validated frames supplied by the shared reader. Keep `render_frame` visible within the module for its test; it need not be a public library API.

- [x] **1. Establish the failing render/encode check.** Add one ignored integration-style unit test in video.rs that constructs a short real trajectory using Task 2, calls record, and uses ffprobe to verify frame count. Ignore it during ordinary Cargo tests because it explicitly requires external ffmpeg; run it during this task and final verification.

```rust
#[test]
#[ignore = "requires ffmpeg and ffprobe"]
fn video_preserves_saved_frame_count_and_size_limit() {
    let dir=std::env::temp_dir().join(format!("md-video-check-{}",std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let config=RunConfig { eq_steps:0,steps:20,sample_every:10,..RunConfig::default() };
    crate::trajectory::write_run(&config,&dir).unwrap();
    let (config,frames)=crate::trajectory::read_run(&dir).unwrap();
    let output=dir.join("run.mp4"); record(&config,&frames,&output).unwrap();
    assert!(std::fs::metadata(&output).unwrap().len()<2_000_000);
    let probe=std::process::Command::new("ffprobe")
        .args(["-v","error","-select_streams","v:0","-count_frames","-show_entries","stream=nb_read_frames","-of","csv=p=0"])
        .arg(&output).output().unwrap();
    assert!(probe.status.success());
    assert_eq!(String::from_utf8(probe.stdout).unwrap().trim(),"2");
    std::fs::remove_dir_all(&dir).unwrap();
}
```

Run `cargo test --manifest-path week2/md/Cargo.toml video_preserves -- --ignored` before implementing record; observe the missing renderer/encoder failure.

- [x] **2. Implement the two-panel frame.** Imports are Plotters prelude, Path, `crate::Result`, `crate::trajectory::{RunConfig,Frame}`, and `crate::analysis::Rdf`. Use 1200×600 pixels with one pixel-per-distance scale for both particle coordinates. Render atom centres, not a false hard-sphere diameter; marker radius is visual only.

```rust
let root=BitMapBackend::new(path,(1200,600)).into_drawing_area(); root.fill(&WHITE)?;
let (atoms,structure)=root.split_horizontally(600);
atoms.draw(&Text::new(format!("Lennard-Jones fluid | t = {:.2}",frame.t),(30,30),("sans-serif",24)))?;
let scale=500.0/config.box_size[0];
let top=(550.0-config.box_size[1]*scale).round() as i32;
atoms.draw(&Rectangle::new([(50,top),(550,550)],BLACK))?;
for pos in &frame.pos {
    atoms.draw(&Circle::new(((50.0+pos[0]*scale).round() as i32,(550.0-pos[1]*scale).round() as i32),3,BLUE.filled()))?;
}
atoms.draw(&Text::new(format!("N = {} | box {:.3} × {:.3}",config.n,config.box_size[0],config.box_size[1]),(50,580),("sans-serif",18)))?;
let mut chart=ChartBuilder::on(&structure)
    .caption("Cumulative radial distribution",("sans-serif",24)).margin(20)
    .x_label_area_size(45).y_label_area_size(65)
    .build_cartesian_2d(0.0..config.box_size[0].min(config.box_size[1])/2.0,0.0..ymax)?;
chart.configure_mesh().max_light_lines(0).x_desc("r / sigma").y_desc("g(r)")
    .axis_desc_style(("sans-serif",18)).label_style(("sans-serif",16)).draw()?;
chart.draw_series(std::iter::once(PathElement::new(vec![(0.0,1.0),(config.box_size[1]/2.0,1.0)],BLACK.mix(0.4))))?;
let points:Vec<_>=rdf.radius.iter().copied().zip(rdf.values.iter().copied()).collect();
chart.draw_series(std::iter::once(PathElement::new(points,RED.stroke_width(2))))?;
atoms.draw(&Text::new(format!("Long-range contrast: {:.4}",rdf.contrast),(30,70),("sans-serif",18)))?;
root.present()?;
Ok(())
```

The lattice geometry guarantees Lx>Ly. Use the same ymax for every frame, computed over all cumulative curves plus the g=1 reference. Visually adjust spacing if the box or titles collide; do not change data to fit the layout.

- [x] **3. Implement bounded-size two-pass encoding.** Use `std::process::Command` arguments, not shell interpolation. Temporary PNGs and x264 pass logs live in a directory created solely by this invocation. Do not delete an unknown pre-existing directory; `create_dir` must fail on collision. Remove the owned directory after successful encoding; on error expose its path for diagnosis.

```rust
let curves=crate::analysis::rdf_series(config,frames);
let ymax=1.08*curves.iter().flat_map(|r| r.values.iter()).copied().fold(1.0_f64,f64::max);
let temp=std::env::temp_dir().join(format!("md-encode-{}",std::process::id()));
std::fs::create_dir(&temp)?;
for (index,(frame,rdf)) in frames.iter().zip(&curves).enumerate() {
    render_frame(config,frame,rdf,ymax,&temp.join(format!("{index:06}.png")))?;
}
let bitrate=(1_600_000_u64*8*30/frames.len() as u64).to_string();
let input=temp.join("%06d.png"); let log=temp.join("x264");
for pass in [1,2] {
    let mut command=std::process::Command::new("ffmpeg");
    command.args(["-hide_banner","-loglevel","error","-y","-framerate","30","-start_number","0","-i"])
        .arg(&input).args(["-an","-c:v","libx264","-pix_fmt","yuv420p","-b:v"])
        .arg(&bitrate).args(["-pass",&pass.to_string(),"-passlogfile"]).arg(&log);
    if pass==1 { command.args(["-f","null","/dev/null"]); }
    else { command.args(["-movflags","+faststart"]).arg(out); }
    let status=command.status()?;
    if !status.success() { return Err(format!("ffmpeg pass {pass} failed; frames at {}",temp.display()).into()); }
}
if std::fs::metadata(out)?.len()>=2_000_000 { return Err(format!("MP4 exceeds size limit; frames at {}",temp.display()).into()); }
std::fs::remove_dir_all(&temp)?;
Ok(())
```

Require the output parent directory to exist; the normal `artifacts/run.mp4` parent is created by run. Do not invent a platform fallback for `/dev/null`; this project's current runtime is macOS. No `-fs` output cap or frame dropping. Both passes consume exactly the same PNG sequence.

- [x] **4. Run the external check and commit.** The full contract video is checked again in Task 5 because it has a different size budget and data range.

```sh
cargo fmt --manifest-path week2/md/Cargo.toml
cargo test --manifest-path week2/md/Cargo.toml video_preserves -- --ignored
git add -- week2/md/src/lib.rs week2/md/src/video.rs
git diff --cached --check
git commit -m "Render saved particle motion with cumulative radial structure"
```

## Task 5: CLI, reproduction, and full contract acceptance

**Files:** Replace `week2/md/src/main.rs`; create `week2/md/tests/cli.rs` and `week2/Makefile`; extend `week2/README.md` and `week2/.gitignore`.

**Interfaces:** CLI `run`, `check`, and `video` call Tasks 2–4 directly. No new library abstraction. `main()->md::Result<()>` prints every physical check before returning an error if the report fails. Clap owns syntax/type errors and unknown flags.

- [x] **1. Add a failing CLI contract test.** Add the parser-equivalence test shown in Step 2 at this stage too, before implementing the parser. The process test checks actual files, positive-only sampling, malformed-file exit status, and unsuccessful physics checks; no source-text assertions or snapshots.

```rust
use std::{fs,process::Command};

#[test]
fn cli_writes_samples_and_reports_failed_or_malformed_input() {
    let bin=env!("CARGO_BIN_EXE_md");
    let dir=std::env::temp_dir().join(format!("md-cli-{}",std::process::id()));
    fs::create_dir(&dir).unwrap();
    let run=Command::new(bin).args(["run","--eq-steps","0","--steps","1000","--sample-every","10","--out"])
        .arg(&dir).output().unwrap();
    assert!(run.status.success(),"{}",String::from_utf8_lossy(&run.stderr));
    let (config,mut frames)=md::trajectory::read_run(&dir).unwrap();
    assert_eq!(frames.len(),100);
    assert_eq!(frames[0].step,10); assert_eq!(frames[99].step,1000);
    assert_eq!(config.n,100); assert_eq!(config.temperature,0.5); assert_eq!(config.seed,2026);
    // Equal speeds intentionally fail the speed-shape gate at a well-defined temperature.
    for frame in &mut frames { for velocity in &mut frame.vel { *velocity=[1.0,0.0]; } }
    let text=frames.iter().map(|f| serde_json::to_string(f).unwrap()+"\n").collect::<String>();
    fs::write(dir.join("traj.jsonl"),text).unwrap();
    let failed=Command::new(bin).arg("check").arg(&dir).output().unwrap();
    assert!(!failed.status.success());
    let report=String::from_utf8(failed.stdout).unwrap();
    for label in ["Secular drift","Temperature","Speed shape","FAIL"] { assert!(report.contains(label)); }
    fs::write(dir.join("traj.jsonl"),"{broken}\n").unwrap();
    assert!(!Command::new(bin).arg("check").arg(&dir).status().unwrap().success());
    assert!(!Command::new(bin).args(["run","--unknown"]).status().unwrap().success());
    fs::remove_dir_all(&dir).unwrap();
}
```

Run `cargo test --manifest-path week2/md/Cargo.toml --test cli` before replacing main; the old binary does not create the required trajectory.

- [x] **2. Implement the CLI.** Use these concrete argument types and direct command routing. Derive `Debug` only where needed by parser tests; do not add dispatch registries.

```rust
use clap::{Args,Parser,Subcommand};
use std::path::PathBuf;
use md::trajectory::{RunConfig,geometry,write_run,read_run};

#[derive(Parser)]
#[command(name="md",about="Lennard-Jones fluid simulation and saved-trajectory analysis")]
struct Cli { #[command(subcommand)] command:Commands }
#[derive(Subcommand)]
enum Commands {
    Run(RunArgs),
    Check { directory:PathBuf },
    Video { directory:PathBuf, #[arg(long)] out:PathBuf },
}
#[derive(Args)]
struct RunArgs {
    #[arg(long,default_value_t=100)] n:usize,
    #[arg(long,default_value_t=0.8)] rho:f64,
    #[arg(long,default_value_t=0.5)] temperature:f64,
    #[arg(long,default_value_t=0.01)] dt:f64,
    #[arg(long,default_value_t=2000)] eq_steps:usize,
    #[arg(long,default_value_t=10000)] steps:usize,
    #[arg(long,default_value_t=50)] sample_every:usize,
    #[arg(long,default_value_t=2026)] seed:u64,
    #[arg(long)] out:PathBuf,
}
fn main()->md::Result<()> {
    match Cli::parse().command {
        Commands::Run(args)=> {
            let config=RunConfig { n:args.n,rho:args.rho,box_size:geometry(args.n,args.rho)?,
                dt:args.dt,temperature:args.temperature,eq_steps:args.eq_steps,steps:args.steps,
                sample_every:args.sample_every,seed:args.seed,integrator:"velocity-verlet".into() };
            write_run(&config,&args.out)?;
            println!("Saved {} frames to {}",config.steps/config.sample_every,args.out.display());
        }
        Commands::Check { directory }=> {
            let (config,frames)=read_run(&directory)?;
            let report=md::analysis::check(&config,&frames)?;
            println!("T_speed = {:.8}; target = {:.8}",report.temperature,config.temperature);
            for (name,value,limit) in [("Secular drift",report.drift,0.002),
                ("Temperature",(report.temperature-config.temperature).abs(),0.05),
                ("Speed shape",report.speed_shape,2.0)] {
                println!("{name}: {value:.8e} < {limit:.8e} {}",if value<limit { "PASS" } else { "FAIL" });
            }
            if !report.passed(config.temperature) { return Err("physics checks failed".into()); }
        }
        Commands::Video { directory,out }=> {
            let (config,frames)=read_run(&directory)?;
            md::video::record(&config,&frames,&out)?;
            println!("Saved {} frames to {}",frames.len(),out.display());
        }
    }
    Ok(())
}
```

Add this parser-level test in main.rs to check the promised equivalence without running the full contract twice:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn abbreviated_run_matches_explicit_contract_flags() {
        let Commands::Run(a)=Cli::try_parse_from(["md","run","--out","artifacts"]).unwrap().command else { panic!() };
        let Commands::Run(b)=Cli::try_parse_from(["md","run","--n","100","--rho","0.8","--temperature","0.5","--dt","0.01","--eq-steps","2000","--steps","10000","--sample-every","50","--seed","2026","--out","artifacts"]).unwrap().command else { panic!() };
        assert_eq!((a.n,a.rho,a.temperature,a.dt,a.eq_steps,a.steps,a.sample_every,a.seed,a.out),
                   (b.n,b.rho,b.temperature,b.dt,b.eq_steps,b.steps,b.sample_every,b.seed,b.out));
    }
}
```

- [x] **3. Add reproduction and user documentation.** Makefile recipe indentation must be a literal tab.

```makefile
.PHONY: reproduce
reproduce:
	cargo run --release --manifest-path md/Cargo.toml -- run --out artifacts
```

Append these entries to week2/.gitignore; keep existing ignores:

```gitignore
/artifacts/
/dimer.png
```

Preserve the README's existing dimer command and add:

```markdown
## Lennard–Jones fluid

From `week2/`:

    make reproduce
    md/target/release/md check artifacts
    md/target/release/md video artifacts --out artifacts/run.mp4

Video requires ffmpeg with libx264. Install ffmpeg first (on macOS, `brew install ffmpeg`).
The default run uses 100 atoms, density 0.8, target temperature 0.5, dt 0.01,
2000 equilibration steps, 10000 production steps, sampling every 50 steps, and seed 2026.
It writes run.json and 200 production frames to traj.jsonl. Production step zero is not saved.

For options, run `md/target/release/md run --help`. To install the `md` command:

    cargo install --path md

The thermostat uses 2N-2 degrees of freedom during equilibration. Saved-speed temperature
uses mean(v²)/2. Check recomputes energy and speed statistics from saved states and exits
unsuccessfully if data are malformed or any of the three physical checks fails.
The video shows the cumulative 100-bin RDF, its long-range contrast, and one image per
saved state at 30 fps. Generated files are excluded from Git.
```

- [x] **4. Run regular checks and the exact default experiment.** Use release mode for the quadratic pair loop. Do not change parameters or random seed to improve acceptance results.

```sh
cargo fmt --manifest-path week2/md/Cargo.toml
cargo fmt --manifest-path week2/md/Cargo.toml -- --check
cargo test --manifest-path week2/md/Cargo.toml
cargo check --manifest-path week2/md/Cargo.toml --examples
make -C week2 reproduce
week2/md/target/release/md check week2/artifacts
week2/md/target/release/md video week2/artifacts --out week2/artifacts/run.mp4
```

If a physical gate fails, inspect force signs, cutoff shift, time stepping, thermostat timing,
normalization, and saved-state analysis. A valid algorithm can still encounter the practical
statistical limit; report the measured failure if diagnosis establishes that, rather than
silently altering the contract or declaring success. Repeat the expensive run only after
changes or an unresolved numerical concern justify it.

- [x] **5. Verify the exact saved-frame and video contracts.**

```sh
python3 - <<'PY'
import json
from pathlib import Path
root=Path('week2/artifacts')
config=json.loads((root/'run.json').read_text())
expected={'n':100,'rho':0.8,'dt':0.01,'temperature':0.5,'eq_steps':2000,'steps':10000,'sample_every':50,'seed':2026,'integrator':'velocity-verlet'}
for key,value in expected.items():
    assert config[key]==value,(key,config[key])
frames=[json.loads(line) for line in (root/'traj.jsonl').read_text().splitlines()]
assert len(frames)==200
for index,frame in enumerate(frames,1):
    assert frame['step']==index*50
    assert frame['t']==frame['step']*config['dt']
    assert len(frame['pos'])==len(frame['vel'])==100
assert (root/'run.mp4').stat().st_size<2_000_000
print('Verified 200 saved frames; MP4 bytes:',(root/'run.mp4').stat().st_size)
PY
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames,r_frame_rate,duration -of json week2/artifacts/run.mp4
ffmpeg -hide_banner -loglevel error -y -i week2/artifacts/run.mp4 -vf 'select=eq(n\,0)+eq(n\,199)' -fps_mode vfr week2/artifacts/inspect-%02d.png
```

Assert ffprobe reports 200 decoded frames at 30/1 fps and approximately 6.67 seconds.
Open both extracted PNGs with the image-viewing tool. Inspect box aspect ratio, current
particle positions, readable labels, a common RDF axis range, and evolving cumulative RDF.
Fix only demonstrated issues, regenerate affected outputs, and inspect again when changed.

- [x] **6. Review scope, mark completed plan steps, and commit the finished tool.** Review all new call sites, ensuring no fixed two-atom loop remains in the shared integrators, no production thermostat is active, and no checker uses stored energy for its verdict. Check that generated outputs and temporary encoding files are absent from the staged diff. Run no unrelated cleanup.

```sh
git check-ignore week2/artifacts/run.json week2/artifacts/traj.jsonl week2/artifacts/run.mp4 week2/dimer.png
git diff --check
git add -- week2/md/src/main.rs week2/md/tests/cli.rs week2/Makefile week2/README.md week2/.gitignore week2/docs/superpowers/plans/2026-09-08-lennard-jones-fluid-cli.md
git diff --cached --stat
git commit -m "Expose fluid CLI and reproducible contract run"
git status --short
```

## Plan self-review

The five tasks cover the shared physical model, every CLI flag/default, initialization and
thermostat schedule, production timing and JSON schema, all three recomputed statistics,
RDF normalization and cumulative contrast, bounded-size encoding, the dimer caller,
reproduction commands, and generated-artifact exclusions. Every cross-task type and
function is named in the producing task's Interfaces section. Unit checks use computed
physical expectations or hand-calculated values; no trajectory snapshots or implementation
freezes are introduced. Execution completed: 16 regular Rust tests passed, and the external ffmpeg test passed separately. Formatting and the dimer example check passed. The exact default run saved 200 frames and passed all three physics gates: drift 0.0000595787566, T_speed 0.5078529535 (deviation 0.0078529535), reduced chi-squared 0.751927273. The video contains 200 decoded frames at 30 fps, lasts 6.666667 seconds, and is 1,561,103 bytes. Early and late decoded frames were visually inspected. Generated outputs are ignored by Git.
