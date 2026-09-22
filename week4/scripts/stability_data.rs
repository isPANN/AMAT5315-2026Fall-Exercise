use std::{f64::consts::TAU, fmt::Write};
use time_stepping::{Integrator, RungeKutta4, advection_diffusion_fourier};

const WIDTH: usize = 480;
const HEIGHT: usize = 640;
const X_MIN: f64 = -3.5;
const X_MAX: f64 = 0.8;
const Y_MIN: f64 = -3.3;
const Y_MAX: f64 = 3.3;
const N: usize = 64;
const END_TIME: f64 = 6.0;

fn write_state(output: &mut String, time: f64, state: &[f64]) {
    write!(output, "{time} ").unwrap();
    for value in state {
        write!(output, "{value} ").unwrap();
    }
    output.push('\n');
}

fn write_trajectory(output: &mut String, step_size: f64) {
    let steps = (END_TIME / step_size).ceil() as usize;
    writeln!(output, "TRAJECTORY {step_size} {} {N}", steps + 1).unwrap();
    let mut state: Vec<_> = (0..N)
        .map(|j| {
            let x = TAU * j as f64 / N as f64;
            let distance = (x - TAU / 4.0 + TAU / 2.0).rem_euclid(TAU) - TAU / 2.0;
            (-0.5 * (distance / 0.35).powi(2)).exp()
        })
        .collect();
    let mut time = 0.0;
    write_state(output, time, &state);
    for _ in 0..steps {
        let step = step_size.min(END_TIME - time);
        state = RungeKutta4.step(&state, step, |values| {
            advection_diffusion_fourier(values, 1.0, 0.05)
        });
        time += step;
        write_state(output, time, &state);
    }
    assert!((time - END_TIME).abs() < f64::EPSILON * END_TIME);
}

fn main() {
    let mut output = format!("{WIDTH} {HEIGHT} {X_MIN} {X_MAX} {Y_MIN} {Y_MAX}\n");
    for row in 0..HEIGHT {
        let imaginary = Y_MIN + (Y_MAX - Y_MIN) * row as f64 / (HEIGHT - 1) as f64;
        for column in 0..WIDTH {
            let real = X_MIN + (X_MAX - X_MIN) * column as f64 / (WIDTH - 1) as f64;
            let advanced = RungeKutta4.step(&[1.0, 0.0], 1.0, |state| {
                vec![
                    real * state[0] - imaginary * state[1],
                    imaginary * state[0] + real * state[1],
                ]
            });
            write!(output, "{} ", advanced[0].hypot(advanced[1])).unwrap();
        }
        output.push('\n');
    }
    write_trajectory(&mut output, 0.045);
    write_trajectory(&mut output, 0.056);
    print!("{output}");
}
