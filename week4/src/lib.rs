use std::f64::consts::TAU;

pub trait Integrator {
    fn step<F>(&self, state: &[f64], step_size: f64, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>;
}

pub struct ForwardEuler;
pub struct ExplicitMidpoint;
pub struct RungeKutta4;
pub struct EqualWeightRungeKutta4;

fn shifted(state: &[f64], rate: &[f64], scale: f64) -> Vec<f64> {
    assert_eq!(state.len(), rate.len(), "rate must match state length");
    state
        .iter()
        .zip(rate)
        .map(|(value, slope)| value + scale * slope)
        .collect()
}

impl Integrator for ForwardEuler {
    fn step<F>(&self, state: &[f64], step_size: f64, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        shifted(state, &rate(state), step_size)
    }
}

impl Integrator for ExplicitMidpoint {
    fn step<F>(&self, state: &[f64], step_size: f64, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let midpoint = shifted(state, &rate(state), step_size / 2.0);
        shifted(state, &rate(&midpoint), step_size)
    }
}

impl Integrator for RungeKutta4 {
    fn step<F>(&self, state: &[f64], step_size: f64, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let k1 = rate(state);
        let k2 = rate(&shifted(state, &k1, step_size / 2.0));
        let k3 = rate(&shifted(state, &k2, step_size / 2.0));
        let k4 = rate(&shifted(state, &k3, step_size));
        assert_eq!(state.len(), k4.len(), "rate must match state length");

        state
            .iter()
            .enumerate()
            .map(|(i, value)| value + step_size * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0)
            .collect()
    }
}

impl Integrator for EqualWeightRungeKutta4 {
    fn step<F>(&self, state: &[f64], step_size: f64, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let k1 = rate(state);
        let k2 = rate(&shifted(state, &k1, step_size / 2.0));
        let k3 = rate(&shifted(state, &k2, step_size / 2.0));
        let k4 = rate(&shifted(state, &k3, step_size));
        assert_eq!(state.len(), k4.len(), "rate must match state length");

        state
            .iter()
            .enumerate()
            .map(|(i, value)| value + step_size * (k1[i] + k2[i] + k3[i] + k4[i]) / 4.0)
            .collect()
    }
}

pub fn advection_diffusion_fourier(state: &[f64], c: f64, nu: f64) -> Vec<f64> {
    let n = state.len();
    assert!(n > 0, "state must not be empty");
    let mut rate = vec![0.0; n];

    // ponytail: direct O(n²) DFT avoids a dependency; use an FFT when large grids demand it.
    for mode in 0..n {
        let wave_number = if mode <= (n - 1) / 2 {
            mode as f64
        } else {
            mode as f64 - n as f64
        };
        let nyquist = n.is_multiple_of(2) && mode == n / 2;
        let mut real = 0.0;
        let mut imaginary = 0.0;
        for (j, &value) in state.iter().enumerate() {
            let angle = TAU * mode as f64 * j as f64 / n as f64;
            real += value * angle.cos();
            imaginary -= value * angle.sin();
        }
        real /= n as f64;
        imaginary /= n as f64;

        let advective_wave_number = if nyquist { 0.0 } else { wave_number };
        let rate_real =
            -nu * wave_number * wave_number * real + c * advective_wave_number * imaginary;
        let rate_imaginary =
            -nu * wave_number * wave_number * imaginary - c * advective_wave_number * real;
        for (j, value) in rate.iter_mut().enumerate() {
            let angle = TAU * mode as f64 * j as f64 / n as f64;
            *value += rate_real * angle.cos() - rate_imaginary * angle.sin();
        }
    }
    rate
}

pub fn advection_diffusion_centered(state: &[f64], c: f64, nu: f64) -> Vec<f64> {
    let n = state.len();
    assert!(n >= 3, "centred differences need at least three points");
    let dx = TAU / n as f64;
    (0..n)
        .map(|i| {
            let left = state[(i + n - 1) % n];
            let right = state[(i + 1) % n];
            -c * (right - left) / (2.0 * dx) + nu * (right - 2.0 * state[i] + left) / (dx * dx)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 64;
    const MODE: f64 = 2.0;
    const C: f64 = 0.7;
    const NU: f64 = 0.03;
    const DT: f64 = 0.02;

    fn wave(time: f64) -> Vec<f64> {
        (0..N)
            .map(|j| {
                let x = TAU * j as f64 / N as f64;
                (-NU * MODE * MODE * time).exp() * (MODE * (x - C * time)).sin()
            })
            .collect()
    }

    fn maximum_error<I: Integrator>(integrator: I) -> f64 {
        let actual = integrator.step(&wave(0.0), DT, |state| {
            advection_diffusion_fourier(state, C, NU)
        });
        actual
            .iter()
            .zip(wave(DT))
            .map(|(actual, exact)| (actual - exact).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn forward_euler_advances_a_single_wave() {
        let error = maximum_error(ForwardEuler);
        assert!(error < 5.0e-4, "maximum error: {error}");
    }

    #[test]
    fn explicit_midpoint_advances_a_single_wave() {
        let error = maximum_error(ExplicitMidpoint);
        assert!(error < 5.0e-6, "maximum error: {error}");
    }

    #[test]
    fn runge_kutta_four_advances_a_single_wave() {
        let error = maximum_error(RungeKutta4);
        assert!(error < 2.0e-10, "maximum error: {error}");
    }

    #[test]
    fn equal_weight_runge_kutta_advances_a_single_wave() {
        let error = maximum_error(EqualWeightRungeKutta4);
        assert!(error < 5.0e-6, "maximum error: {error}");
    }

    #[test]
    fn nyquist_mode_does_not_advect() {
        let state: Vec<_> = (0..8)
            .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let rate = advection_diffusion_fourier(&state, 1.0, 0.0);
        assert!(rate.iter().all(|value| value.abs() < 1.0e-12));
    }

    #[test]
    fn centred_rate_matches_a_single_wave() {
        let state = wave(0.0);
        let actual = advection_diffusion_centered(&state, C, NU);
        let exact: Vec<_> = (0..N)
            .map(|j| {
                let x = TAU * j as f64 / N as f64;
                -C * MODE * (MODE * x).cos() - NU * MODE * MODE * (MODE * x).sin()
            })
            .collect();
        let error = actual
            .iter()
            .zip(exact)
            .map(|(actual, exact)| (actual - exact).abs())
            .fold(0.0, f64::max);
        assert!(error < 1.0e-2);
    }
}
