use std::{f64::consts::TAU, fmt::Write};
use time_stepping::{
    ForwardEuler, Integrator, RungeKutta4, advection_diffusion_centered,
    advection_diffusion_fourier,
};

const N: usize = 64;

fn initial_state() -> Vec<f64> {
    (0..N)
        .map(|j| {
            let x = TAU * j as f64 / N as f64;
            (-3..=3)
                .map(|image| {
                    let distance = x - TAU / 4.0 + image as f64 * TAU;
                    (-0.5 * (distance / 0.25).powi(2)).exp()
                })
                .sum()
        })
        .collect()
}

fn integrate<I, F>(integrator: I, step_size: f64, rate: F) -> Vec<f64>
where
    I: Integrator,
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut state = initial_state();
    let mut time = 0.0;
    while time < TAU {
        let step = step_size.min(TAU - time);
        state = integrator.step(&state, step, &rate);
        time += step;
    }
    assert!((time - TAU).abs() < f64::EPSILON * TAU);
    state
}

fn write_run(output: &mut String, name: &str, state: &[f64]) {
    writeln!(output, "{name}").unwrap();
    for value in state {
        write!(output, "{value} ").unwrap();
    }
    output.push('\n');
}

fn main() {
    let mut output = format!("{N}\n");
    write_run(
        &mut output,
        "rk4-fourier",
        &integrate(RungeKutta4, 0.02, |state| {
            advection_diffusion_fourier(state, 1.0, 0.002)
        }),
    );
    write_run(
        &mut output,
        "rk4-centered",
        &integrate(RungeKutta4, 0.02, |state| {
            advection_diffusion_centered(state, 1.0, 0.002)
        }),
    );
    write_run(
        &mut output,
        "euler-fourier",
        &integrate(ForwardEuler, 0.005, |state| {
            advection_diffusion_fourier(state, 1.0, 0.002)
        }),
    );
    print!("{output}");
}
