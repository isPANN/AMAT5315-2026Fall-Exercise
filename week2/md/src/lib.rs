pub mod analysis;
pub mod trajectory;
pub mod video;
pub fn greeting() -> &'static str {
    "Hello, world!"
}

pub fn lennard_jones_energy(r: f64) -> f64 {
    4.0 * (r.powi(-12) - r.powi(-6))
}

pub fn lennard_jones_force(r: f64) -> f64 {
    24.0 * (2.0 * r.powi(-13) - r.powi(-7))
}

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const RC: f64 = 2.5;

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Force {
    Naive,
    Cells,
}

#[derive(Clone, Debug)]
pub struct State {
    pub positions: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    pub box_size: Option<[f64; 2]>,
    pub force: Force,
}
impl State {
    pub fn displacement(&self, i: usize, j: usize) -> [f64; 2] {
        let mut d = std::array::from_fn(|axis| self.positions[j][axis] - self.positions[i][axis]);
        if let Some(lengths) = self.box_size {
            for axis in 0..2 {
                d[axis] -= lengths[axis] * (d[axis] / lengths[axis]).round();
            }
        }
        d
    }

    pub fn interactions(&self) -> Result<(Vec<[f64; 2]>, f64)> {
        match self.force {
            Force::Naive => self.interactions_naive(),
            Force::Cells => self.interactions_cells(),
        }
    }

    fn add_pair(&self, i: usize, j: usize, forces: &mut [[f64; 2]]) -> Result<f64> {
        let d = self.displacement(i, j);
        let r2 = d[0] * d[0] + d[1] * d[1];
        if !r2.is_finite() || r2 <= 0.0 {
            return Err(format!("invalid separation for atoms {i}, {j}").into());
        }
        if self.box_size.is_some() && r2 >= RC * RC {
            return Ok(0.0);
        }
        let r = r2.sqrt();
        let u = lennard_jones_energy(r);
        let scale = lennard_jones_force(r) / r;
        if !u.is_finite() || !scale.is_finite() {
            return Err(format!("nonfinite interaction for atoms {i}, {j}").into());
        }
        for axis in 0..2 {
            forces[i][axis] -= scale * d[axis];
            forces[j][axis] += scale * d[axis];
        }
        Ok(u - if self.box_size.is_some() {
            lennard_jones_energy(RC)
        } else {
            0.0
        })
    }

    pub fn interactions_naive(&self) -> Result<(Vec<[f64; 2]>, f64)> {
        let mut forces = vec![[0.0; 2]; self.positions.len()];
        let mut potential = 0.0;
        for i in 0..self.positions.len() {
            for j in i + 1..self.positions.len() {
                potential += self.add_pair(i, j, &mut forces)?;
            }
        }
        Ok((forces, potential))
    }

    pub fn interactions_cells(&self) -> Result<(Vec<[f64; 2]>, f64)> {
        let lengths = self.box_size.ok_or("cell forces require a periodic box")?;
        if lengths.iter().any(|l| !l.is_finite() || *l < RC) {
            return Err("cell box sides must be finite and at least rc".into());
        }
        let counts = lengths.map(|l| (l / RC).floor() as usize);
        let widths: [f64; 2] = std::array::from_fn(|axis| lengths[axis] / counts[axis] as f64);
        let mut cells = vec![Vec::new(); counts[0] * counts[1]];
        for (i, position) in self.positions.iter().enumerate() {
            if position.iter().any(|v| !v.is_finite()) {
                return Err(format!("nonfinite position for atom {i}").into());
            }
            let index: [usize; 2] = std::array::from_fn(|axis| {
                (position[axis].rem_euclid(lengths[axis]) / widths[axis]).floor() as usize
                    % counts[axis]
            });
            cells[index[0] + counts[0] * index[1]].push(i);
        }
        let mut forces = vec![[0.0; 2]; self.positions.len()];
        let mut potential = 0.0;
        for (cell, atoms) in cells.iter().enumerate() {
            let x = cell % counts[0];
            let y = cell / counts[0];
            let mut neighbours = [0; 9];
            let mut used = 0;
            for dy in [counts[1] - 1, 0, 1] {
                for dx in [counts[0] - 1, 0, 1] {
                    let neighbour = (x + dx) % counts[0] + counts[0] * ((y + dy) % counts[1]);
                    if !neighbours[..used].contains(&neighbour) {
                        neighbours[used] = neighbour;
                        used += 1;
                    }
                }
            }
            for &i in atoms {
                for &neighbour in &neighbours[..used] {
                    for &j in &cells[neighbour] {
                        if j > i {
                            potential += self.add_pair(i, j, &mut forces)?;
                        }
                    }
                }
            }
        }
        Ok((forces, potential))
    }

    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.velocities.iter().flatten().map(|v| v * v).sum::<f64>()
    }

    pub fn energy(&self) -> Result<f64> {
        let energy = self.interactions()?.1 + self.kinetic_energy();
        if !energy.is_finite() {
            return Err("nonfinite total energy".into());
        }
        Ok(energy)
    }
}

pub trait Integrator {
    fn step(&self, state: &mut State, dt: f64) -> Result<()>;
}
pub struct ForwardEuler;
pub struct VelocityVerlet;
impl Integrator for ForwardEuler {
    fn step(&self, state: &mut State, dt: f64) -> Result<()> {
        let acceleration = state.interactions()?.0;
        for (atom, force) in acceleration.iter().enumerate() {
            for axis in 0..2 {
                state.positions[atom][axis] += dt * state.velocities[atom][axis];
                state.velocities[atom][axis] += dt * force[axis];
                if let Some(lengths) = state.box_size {
                    state.positions[atom][axis] =
                        state.positions[atom][axis].rem_euclid(lengths[axis]);
                }
            }
        }
        Ok(())
    }
}
impl Integrator for VelocityVerlet {
    fn step(&self, state: &mut State, dt: f64) -> Result<()> {
        let old = state.interactions()?.0;
        for (atom, acceleration) in old.iter().enumerate() {
            for axis in 0..2 {
                state.positions[atom][axis] +=
                    dt * state.velocities[atom][axis] + 0.5 * dt * dt * acceleration[axis];
                if let Some(lengths) = state.box_size {
                    state.positions[atom][axis] =
                        state.positions[atom][axis].rem_euclid(lengths[axis]);
                }
            }
        }
        let new = state.interactions()?.0;
        for atom in 0..state.positions.len() {
            for axis in 0..2 {
                state.velocities[atom][axis] += 0.5 * dt * (old[atom][axis] + new[atom][axis]);
            }
        }
        Ok(())
    }
}
pub fn dimer_state() -> State {
    State {
        positions: vec![[0.0, 0.0], [1.2, 0.0]],
        velocities: vec![[0.0, 0.0]; 2],
        box_size: None,
        force: Force::Naive,
    }
}
pub struct Sample {
    pub step: usize,
    pub time: f64,
    pub total_energy: f64,
    pub energy_error: f64,
}
pub fn run(
    integrator: &impl Integrator,
    mut state: State,
    dt: f64,
    steps: usize,
) -> Result<Vec<Sample>> {
    let initial_energy = state.energy()?;
    let mut samples = Vec::with_capacity(steps + 1);
    for step in 0..=steps {
        if step > 0 {
            integrator.step(&mut state, dt)?;
        }
        let total_energy = state.energy()?;
        samples.push(Sample {
            step,
            time: step as f64 * dt,
            total_energy,
            energy_error: total_energy - initial_energy,
        });
    }
    Ok(samples)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_match_naive_on_perturbed_lattices_and_periodic_edge_pairs() {
        fn compare(state: &State) {
            let (naive, naive_energy) = state.interactions_naive().unwrap();
            let (cells, cells_energy) = state.interactions_cells().unwrap();
            for (a, b) in naive.iter().flatten().zip(cells.iter().flatten()) {
                assert!((a - b).abs() <= 1e-10 * (1.0 + a.abs()), "{a} != {b}");
            }
            assert!(
                (naive_energy - cells_energy).abs() <= 1e-10 * (1.0 + naive_energy.abs()),
                "{naive_energy} != {cells_energy}"
            );
        }
        for (n, rho) in [(16, 0.5), (100, 0.8), (400, 0.8)] {
            let config = trajectory::RunConfig {
                n,
                rho,
                box_size: trajectory::geometry(n, rho).unwrap(),
                ..trajectory::RunConfig::default()
            };
            if n == 16 {
                assert_eq!(config.box_size.map(|l| (l / RC).floor() as usize), [2, 2]);
            }
            let mut state = trajectory::initial_state(&config).unwrap();
            for shift in 0..3 {
                for (i, position) in state.positions.iter_mut().enumerate() {
                    for axis in 0..2 {
                        position[axis] = (position[axis]
                            + 0.08 * ((i * 7 + axis * 3 + shift) as f64).sin()
                            + 0.7 * config.box_size[axis])
                            .rem_euclid(config.box_size[axis]);
                    }
                }
                compare(&state);
            }
        }
        for lengths in [[6.0, 6.0], [6.0, 12.0], [12.0, 9.0]] {
            let mut state = State {
                positions: vec![[0.0, 0.0]; 2],
                velocities: vec![[0.0; 2]; 2],
                box_size: Some(lengths),
                force: Force::Cells,
            };
            for pair in [
                [[0.1, 1.0], [lengths[0] - 1.0, 1.0]],
                [[1.0, 0.1], [1.0, lengths[1] - 1.0]],
                [[0.1, 0.1], [lengths[0] - 0.7, lengths[1] - 0.7]],
                [[0.0, 0.0], [RC - 1e-10, 0.0]],
                [[0.0, 0.0], [RC, 0.0]],
                [[0.0, 0.0], [RC + 1e-10, 0.0]],
            ] {
                state.positions = pair.to_vec();
                compare(&state);
                if pair[1][0] == RC || pair[1][0] == RC + 1e-10 {
                    assert_eq!(state.interactions().unwrap(), (vec![[0.0; 2]; 2], 0.0));
                }
            }
            state.positions = vec![[0.0, 0.0]; 2];
            assert!(state.interactions().is_err());
            state.positions[1][0] = f64::NAN;
            assert!(state.interactions().is_err());
        }
        let mut dimer = dimer_state();
        dimer.force = Force::Cells;
        assert!(dimer.interactions().is_err());
    }

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }

    #[test]
    fn lennard_jones_energy_at_two() {
        assert_eq!(super::lennard_jones_energy(2.0), -0.0615234375);
    }

    #[test]
    fn lennard_jones_force_at_two() {
        assert_eq!(super::lennard_jones_force(2.0), -0.181640625);
    }

    #[test]
    fn pair_force_and_total_energy() {
        let state = State {
            positions: vec![[0.0, 0.0], [0.72, 0.96]],
            velocities: vec![[1.0, 2.0], [-3.0, 4.0]],
            box_size: None,
            force: Force::Naive,
        };
        let force = state.interactions().unwrap().0;
        // Separation 1.2 along (0.6, 0.8); kinetic energy is 15.
        assert!((force[1][0] + 1.327016005333847).abs() < 1e-12);
        assert!((force[1][1] + 1.769354673778463).abs() < 1e-12);
        for axis in 0..2 {
            assert_eq!(force[0][axis], -force[1][axis]);
        }
        assert!((state.energy().unwrap() - 14.109034712416924).abs() < 1e-12);
    }

    #[test]
    fn integrator_steps_follow_their_formulas() {
        let initial = State {
            positions: vec![[0.0, 0.0], [1.2, 0.0]],
            velocities: vec![[0.1, 0.2], [-0.1, -0.2]],
            box_size: None,
            force: Force::Naive,
        };
        let mut euler = initial.clone();
        ForwardEuler.step(&mut euler, 0.01).unwrap();
        let mut verlet = initial;
        VelocityVerlet.step(&mut verlet, 0.01).unwrap();
        // Independently evaluated one-step formulas, including transverse motion.
        for (actual, positions, velocities) in [
            (
                euler,
                [[0.001, 0.002], [1.199, -0.002]],
                [[0.12211693342223078, 0.2], [-0.12211693342223078, -0.2]],
            ),
            (
                verlet,
                [[0.001110584667111154, 0.002], [1.198889415332889, -0.002]],
                [
                    [0.1220075492806964, 0.19996343537792033],
                    [-0.1220075492806964, -0.19996343537792033],
                ],
            ),
        ] {
            for atom in 0..2 {
                for axis in 0..2 {
                    assert!((actual.positions[atom][axis] - positions[atom][axis]).abs() < 1e-12);
                    assert!((actual.velocities[atom][axis] - velocities[atom][axis]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn experiment_sampling_and_energy_behavior() {
        fn check(method: &impl Integrator, steps: usize) -> Vec<Sample> {
            let samples = run(method, dimer_state(), 0.01, steps).unwrap();
            assert_eq!(samples.len(), steps + 1);
            assert_eq!(samples[0].step, 0);
            assert_eq!(samples[0].time, 0.0);
            // Independently evaluated floating-point energies can differ by roundoff.
            let energy_tolerance = 1e-12;
            assert!((samples[0].total_energy - lennard_jones_energy(1.2)).abs() < energy_tolerance);
            assert_eq!(samples[0].energy_error, 0.0);
            assert_eq!(samples[steps].step, steps);
            assert_eq!(samples[steps].time, steps as f64 * 0.01);
            let mut state = dimer_state();
            for (step, sample) in samples.iter().enumerate() {
                if step > 0 {
                    method.step(&mut state, 0.01).unwrap();
                }
                assert_eq!(sample.step, step);
                assert_eq!(sample.time, step as f64 * 0.01);
                let energy = state.energy().unwrap();
                assert!((sample.total_energy - energy).abs() < energy_tolerance);
                assert!(
                    (sample.energy_error - (energy - lennard_jones_energy(1.2))).abs()
                        < energy_tolerance
                );
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
            samples
                .iter()
                .map(|s| s.energy_error.abs())
                .fold(0.0_f64, f64::max)
        };
        assert!(max_error(&verlet[..=500]) < max_error(&euler));
        // Acceptance threshold: one percent of the initial energy magnitude.
        assert!(max_error(&verlet) < 0.01 * lennard_jones_energy(1.2).abs());
    }

    #[test]
    fn experiment_rejects_nonfinite_energy() {
        let state = State {
            positions: vec![[0.0, 0.0]; 2],
            velocities: vec![[0.0, 0.0]; 2],
            box_size: None,
            force: Force::Naive,
        };
        assert!(run(&ForwardEuler, state, 0.01, 1).is_err());
    }

    #[test]
    fn periodic_pair_uses_shifted_energy_and_plain_force() {
        let mut state = State {
            positions: vec![[0.0, 0.0], [5.0, 0.0]],
            velocities: vec![[0.0, 0.0]; 2],
            box_size: Some([6.0, 6.0]),
            force: Force::Naive,
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
}
