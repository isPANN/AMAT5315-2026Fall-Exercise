pub fn greeting() -> &'static str {
    "Hello, world!"
}

pub fn lennard_jones_energy(r: f64) -> f64 {
    4.0 * (r.powi(-12) - r.powi(-6))
}

pub fn lennard_jones_force(r: f64) -> f64 {
    24.0 * (2.0 * r.powi(-13) - r.powi(-7))
}

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
                state.positions[atom][axis] +=
                    dt * state.velocities[atom][axis] + 0.5 * dt * dt * a;
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
            positions: [[0.0, 0.0], [0.72, 0.96]],
            velocities: [[1.0, 2.0], [-3.0, 4.0]],
        };
        let force = state.forces();
        // Separation 1.2 along (0.6, 0.8); kinetic energy is 15.
        assert!((force[1][0] + 1.327016005333847).abs() < 1e-12);
        assert!((force[1][1] + 1.769354673778463).abs() < 1e-12);
        for axis in 0..2 {
            assert_eq!(force[0][axis], -force[1][axis]);
        }
        assert!((state.energy() - 14.109034712416924).abs() < 1e-12);
    }

    #[test]
    fn integrator_steps_follow_their_formulas() {
        let initial = State {
            positions: [[0.0, 0.0], [1.2, 0.0]],
            velocities: [[0.1, 0.2], [-0.1, -0.2]],
        };
        let mut euler = initial;
        ForwardEuler.step(&mut euler, 0.01);
        let mut verlet = initial;
        VelocityVerlet.step(&mut verlet, 0.01);
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
}
