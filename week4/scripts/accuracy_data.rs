use std::{f64::consts::TAU, fmt::Write};
use time_stepping::{
    EqualWeightRungeKutta4, ExplicitMidpoint, ForwardEuler, Integrator, RungeKutta4,
    advection_diffusion_centered, advection_diffusion_fourier,
};

const N: usize = 64;
const STEPS: [f64; 4] = [0.02, 0.01, 0.005, 0.0025];

fn initial_state(sigma: f64) -> Vec<f64> {
    (0..N)
        .map(|j| {
            let x = TAU * j as f64 / N as f64;
            (-3..=3)
                .map(|image| {
                    let distance = x - TAU / 4.0 + image as f64 * TAU;
                    (-0.5 * (distance / sigma).powi(2)).exp()
                })
                .sum()
        })
        .collect()
}

fn integrate<I, F>(integrator: &I, step_size: f64, end_time: f64, sigma: f64, rate: &F) -> Vec<f64>
where
    I: Integrator,
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut state = initial_state(sigma);
    let mut time = 0.0;
    while time < end_time {
        let step = step_size.min(end_time - time);
        state = integrator.step(&state, step, rate);
        time += step;
    }
    assert!((time - end_time).abs() < f64::EPSILON * end_time);
    state
}

fn write_run(output: &mut String, name: &str, state: &[f64]) {
    writeln!(output, "{name}").unwrap();
    for value in state {
        write!(output, "{value} ").unwrap();
    }
    output.push('\n');
}

fn write_convergence<I, F>(output: &mut String, name: &str, integrator: &I, rate: F)
where
    I: Integrator,
    F: Fn(&[f64]) -> Vec<f64>,
{
    for step in STEPS {
        write_run(
            output,
            &format!("CONVERGENCE {name} {step}"),
            &integrate(integrator, step, 1.0, 0.35, &rate),
        );
    }
}

fn main() {
    let mut output = format!("{N}\n");
    write_run(
        &mut output,
        "rk4-fourier",
        &integrate(&RungeKutta4, 0.02, TAU, 0.25, &|state| {
            advection_diffusion_fourier(state, 1.0, 0.002)
        }),
    );
    write_run(
        &mut output,
        "rk4-centered",
        &integrate(&RungeKutta4, 0.02, TAU, 0.25, &|state| {
            advection_diffusion_centered(state, 1.0, 0.002)
        }),
    );
    write_run(
        &mut output,
        "euler-fourier",
        &integrate(&ForwardEuler, 0.005, TAU, 0.25, &|state| {
            advection_diffusion_fourier(state, 1.0, 0.002)
        }),
    );
    let rate = |state: &[f64]| advection_diffusion_fourier(state, 1.0, 0.05);
    write_convergence(&mut output, "Euler", &ForwardEuler, rate);
    write_convergence(&mut output, "Midpoint", &ExplicitMidpoint, rate);
    write_convergence(&mut output, "RK4", &RungeKutta4, rate);
    write_convergence(
        &mut output,
        "Equal-weight-RK4",
        &EqualWeightRungeKutta4,
        rate,
    );
    print!("{output}");
}
