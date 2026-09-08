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

#[derive(Clone, Debug)]
pub struct State {
    pub positions: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    pub box_size: Option<[f64; 2]>,
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
        let mut forces = vec![[0.0; 2]; self.positions.len()];
        let mut potential = 0.0;
        // ponytail: O(N²) pair loop; use cell lists if larger-run timings require them.
        for i in 0..self.positions.len() {
            for j in i + 1..self.positions.len() {
                let d = self.displacement(i, j);
                let r2 = d[0] * d[0] + d[1] * d[1];
                if !r2.is_finite() || r2 <= 0.0 {
                    return Err(format!("invalid separation for atoms {i}, {j}").into());
                }
                if self.box_size.is_some() && r2 >= RC * RC {
                    continue;
                }
                let r = r2.sqrt();
                let u = lennard_jones_energy(r);
                let scale = lennard_jones_force(r) / r;
                if !u.is_finite() || !scale.is_finite() {
                    return Err(format!("nonfinite interaction for atoms {i}, {j}").into());
                }
                potential += u - if self.box_size.is_some() {
                    lennard_jones_energy(RC)
                } else {
                    0.0
                };
                for axis in 0..2 {
                    forces[i][axis] -= scale * d[axis];
                    forces[j][axis] += scale * d[axis];
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
            assert_eq!(samples[0].total_energy, lennard_jones_energy(1.2));
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
                assert_eq!(sample.total_energy, state.energy().unwrap());
                assert_eq!(
                    sample.energy_error,
                    state.energy().unwrap() - lennard_jones_energy(1.2)
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
        };
        assert!(run(&ForwardEuler, state, 0.01, 1).is_err());
    }

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
}
