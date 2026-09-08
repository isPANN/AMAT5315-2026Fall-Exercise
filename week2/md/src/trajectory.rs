use crate::{Integrator, RC, Result, State, VelocityVerlet};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use std::{
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RunConfig {
    pub n: usize,
    pub rho: f64,
    #[serde(rename = "box")]
    pub box_size: [f64; 2],
    pub dt: f64,
    pub temperature: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub integrator: String,
}
impl Default for RunConfig {
    fn default() -> Self {
        Self {
            n: 100,
            rho: 0.8,
            box_size: geometry(100, 0.8).unwrap(),
            dt: 0.01,
            temperature: 0.5,
            eq_steps: 2000,
            steps: 10000,
            sample_every: 50,
            seed: 2026,
            integrator: "velocity-verlet".into(),
        }
    }
}
pub fn geometry(n: usize, rho: f64) -> Result<[f64; 2]> {
    let q = n.isqrt();
    if q < 2 || q * q != n || q % 2 != 0 {
        return Err("n must be an even square grid".into());
    }
    if !rho.is_finite() || rho <= 0.0 {
        return Err("rho must be finite and positive".into());
    }
    let a = (2.0 / (3.0_f64.sqrt() * rho)).sqrt();
    let lengths = [q as f64 * a, q as f64 * a * 3.0_f64.sqrt() / 2.0];
    if lengths.iter().any(|v| !v.is_finite() || *v <= 2.0 * RC) {
        return Err("box sides must exceed 2*rc".into());
    }
    Ok(lengths)
}
pub fn rescale(state: &mut State, target: f64) -> Result<()> {
    let t = 2.0 * state.kinetic_energy() / (2 * state.velocities.len() - 2) as f64;
    if !t.is_finite() || t <= 0.0 {
        return Err("undefined thermostat temperature".into());
    }
    let scale = (target / t).sqrt();
    for v in state.velocities.iter_mut().flatten() {
        *v *= scale;
    }
    Ok(())
}
impl RunConfig {
    pub fn validate(&self) -> Result<()> {
        let expected = geometry(self.n, self.rho)?;
        for axis in 0..2 {
            if !self.box_size[axis].is_finite()
                || (self.box_size[axis] - expected[axis]).abs()
                    > 1e-12 * expected[axis].abs().max(1.0)
            {
                return Err("box does not match n and rho".into());
            }
        }
        for (field, value) in [("dt", self.dt), ("temperature", self.temperature)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{field} must be finite and positive").into());
            }
        }
        if self.sample_every == 0 || self.steps < self.sample_every {
            return Err("run must save at least one production frame".into());
        }
        if !(self.steps as f64 * self.dt).is_finite() {
            return Err("nonfinite production duration".into());
        }
        if self.integrator != "velocity-verlet" {
            return Err("unsupported integrator".into());
        }
        Ok(())
    }
}
pub fn initial_state(config: &RunConfig) -> Result<State> {
    config.validate()?;
    let q = config.n.isqrt();
    let a = config.box_size[0] / q as f64;
    let h = config.box_size[1] / q as f64;
    let positions = (0..q)
        .flat_map(|j| (0..q).map(move |i| [(i as f64 + 0.5 * (j % 2) as f64) * a, j as f64 * h]))
        .collect();
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let normal = Normal::new(0.0, config.temperature.sqrt())?;
    let mut velocities: Vec<[f64; 2]> = (0..config.n)
        .map(|_| [normal.sample(&mut rng), normal.sample(&mut rng)])
        .collect();
    for axis in 0..2 {
        let mean = velocities.iter().map(|v| v[axis]).sum::<f64>() / config.n as f64;
        for v in &mut velocities {
            v[axis] -= mean;
        }
    }
    let mut state = State {
        positions,
        velocities,
        box_size: Some(config.box_size),
    };
    rescale(&mut state, config.temperature)?;
    Ok(state)
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Frame {
    pub step: usize,
    pub t: f64,
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    #[serde(rename = "E_pot")]
    pub e_pot: f64,
    #[serde(rename = "E_kin")]
    pub e_kin: f64,
}
impl Frame {
    pub fn state(&self, config: &RunConfig) -> State {
        State {
            positions: self.pos.clone(),
            velocities: self.vel.clone(),
            box_size: Some(config.box_size),
        }
    }
}
impl Frame {
    pub fn validate(&self, config: &RunConfig, index: usize) -> Result<()> {
        if self.step != (index + 1) * config.sample_every {
            return Err("invalid production sampling sequence".into());
        }
        let expected = self.step as f64 * config.dt;
        if !self.t.is_finite() || (self.t - expected).abs() > 1e-12 * expected.abs().max(1.0) {
            return Err("t does not equal step*dt".into());
        }
        if self.pos.len() != config.n || self.vel.len() != config.n {
            return Err("particle count mismatch".into());
        }
        if self
            .pos
            .iter()
            .chain(&self.vel)
            .flatten()
            .any(|v| !v.is_finite())
            || !self.e_pot.is_finite()
            || !self.e_kin.is_finite()
        {
            return Err("nonfinite frame data".into());
        }
        for p in &self.pos {
            for axis in 0..2 {
                if p[axis] < 0.0 || p[axis] >= config.box_size[axis] {
                    return Err("position is outside the wrapped box".into());
                }
            }
        }
        Ok(())
    }
}
pub fn write_run(config: &RunConfig, out: &Path) -> Result<()> {
    let mut state = initial_state(config)?;
    std::fs::create_dir_all(out)?;
    let mut metadata = BufWriter::new(std::fs::File::create(out.join("run.json"))?);
    serde_json::to_writer_pretty(&mut metadata, config)?;
    metadata.flush()?;
    let mut file = BufWriter::new(std::fs::File::create(out.join("traj.jsonl"))?);
    for step in 1..=config.eq_steps {
        VelocityVerlet
            .step(&mut state, config.dt)
            .map_err(|e| format!("equilibration step {step}: {e}"))?;
        if step % 50 == 0 {
            rescale(&mut state, config.temperature)?;
        }
    }
    for step in 1..=config.steps {
        VelocityVerlet
            .step(&mut state, config.dt)
            .map_err(|e| format!("production step {step}: {e}"))?;
        if step % config.sample_every == 0 {
            let frame = Frame {
                step,
                t: step as f64 * config.dt,
                pos: state.positions.clone(),
                vel: state.velocities.clone(),
                e_pot: state.interactions()?.1,
                e_kin: state.kinetic_energy(),
            };
            frame.validate(config, step / config.sample_every - 1)?;
            serde_json::to_writer(&mut file, &frame)?;
            writeln!(file)?;
        }
    }
    file.flush()?;
    Ok(())
}
pub fn read_run(dir: &Path) -> Result<(RunConfig, Vec<Frame>)> {
    let config: RunConfig =
        serde_json::from_reader(BufReader::new(std::fs::File::open(dir.join("run.json"))?))?;
    config.validate()?;
    let mut frames = Vec::new();
    for (index, line) in BufReader::new(std::fs::File::open(dir.join("traj.jsonl"))?)
        .lines()
        .enumerate()
    {
        let frame: Frame =
            serde_json::from_str(&line?).map_err(|e| format!("frame {}: {e}", index + 1))?;
        frame
            .validate(&config, index)
            .map_err(|e| format!("frame {}: {e}", index + 1))?;
        frame
            .state(&config)
            .energy()
            .map_err(|e| format!("frame {}: {e}", index + 1))?;
        frames.push(frame);
    }
    if frames.len() != config.steps / config.sample_every {
        return Err("trajectory frame count does not match run.json".into());
    }
    Ok((config, frames))
}
#[cfg(test)]
mod tests {
    use super::*;
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
        assert!((lengths[0] * lengths[1] - 125.0).abs() < 1e-10);
        assert!((a.positions[10][0] - a.positions[1][0] / 2.0).abs() < 1e-12);
        for n in [100, 400, 1600] {
            let lengths = geometry(n, 0.8).unwrap();
            assert!((lengths[0] * lengths[1] - n as f64 / 0.8).abs() < 1e-9);
        }
        assert!(geometry(99, 0.8).is_err());
        assert!(geometry(81, 0.8).is_err());
    }
    #[test]
    fn production_files_exclude_zero_and_validate_sampling() {
        let dir = std::env::temp_dir().join(format!("md-production-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let config = RunConfig {
            eq_steps: 51,
            steps: 61,
            sample_every: 10,
            ..RunConfig::default()
        };
        write_run(&config, &dir).unwrap();
        let (saved, frames) = read_run(&dir).unwrap();
        assert_eq!(saved.eq_steps, 51);
        assert_eq!(
            frames.iter().map(|f| f.step).collect::<Vec<_>>(),
            vec![10, 20, 30, 40, 50, 60]
        );
        assert_eq!(frames[0].t, 0.1);
        assert_eq!(frames[5].t, 0.6);
        let mut manual = initial_state(&config).unwrap();
        for step in 1..=111 {
            VelocityVerlet.step(&mut manual, config.dt).unwrap();
            if step == 50 {
                rescale(&mut manual, config.temperature).unwrap();
            }
        }
        assert_eq!(frames[5].pos, manual.positions);
        assert_eq!(frames[5].vel, manual.velocities);
        let mut bad = frames[0].clone();
        bad.step = 0;
        assert!(bad.validate(&config, 0).is_err());
        bad = frames[0].clone();
        bad.pos[0][0] = -0.1;
        assert!(bad.validate(&config, 0).is_err());
        std::fs::write(dir.join("traj.jsonl"), "{bad json}\n").unwrap();
        assert!(read_run(&dir).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
