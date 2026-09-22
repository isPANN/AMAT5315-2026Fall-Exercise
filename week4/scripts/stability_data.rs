use std::fmt::Write;
use time_stepping::{Integrator, RungeKutta4};

const WIDTH: usize = 480;
const HEIGHT: usize = 640;
const X_MIN: f64 = -3.5;
const X_MAX: f64 = 0.8;
const Y_MIN: f64 = -3.3;
const Y_MAX: f64 = 3.3;

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
    print!("{output}");
}
