use serde::Serialize;
use std::{
    env,
    f64::consts::TAU,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::Path,
    process,
};
use time_stepping::{
    Integrator, RungeKutta4,
    vorticity::{SpectralGrid, VelocityField},
};

const DT: f64 = 0.01;
const STEPS: usize = 2000;
const SNAPSHOT_STEPS: usize = 50;

#[derive(Serialize)]
struct Snapshot<'a> {
    t: f64,
    omega: &'a [f64],
}

fn write_snapshot(writer: &mut impl Write, time: f64, omega: &[f64]) -> Result<(), String> {
    serde_json::to_writer(writer.by_ref(), &Snapshot { t: time, omega })
        .map_err(|error| error.to_string())?;
    writer.write_all(b"\n").map_err(|error| error.to_string())
}

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: sensitivity-data <nu> <output-directory>".into());
    }
    let nu: f64 = args[0].parse().map_err(|_| "invalid viscosity")?;
    if !nu.is_finite() || nu < 0.0 {
        return Err("viscosity must be nonnegative and finite".into());
    }
    let field: VelocityField =
        serde_json::from_reader(io::stdin().lock()).map_err(|error| error.to_string())?;
    let size = field.n.checked_mul(field.n).ok_or("n is too large")?;
    if field.u.len() != size || field.v.len() != size {
        return Err("u and v must each contain n*n values".into());
    }

    let grid = SpectralGrid::new(field.n);
    let mut original = grid.vorticity_from_velocity(&field.u, &field.v);
    let mut perturbed = original.clone();
    let largest_component = field
        .u
        .iter()
        .chain(field.v.iter())
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    for y_index in 0..field.n {
        let y = TAU * y_index as f64 / field.n as f64;
        for x_index in 0..field.n {
            let x = TAU * x_index as f64 / field.n as f64;
            perturbed[y_index * field.n + x_index] +=
                -7e-5 * largest_component * (3.0 * x).cos() * (4.0 * y).cos();
        }
    }
    perturbed = grid.filter(&perturbed);

    let output = Path::new(&args[1]);
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    let mut original_file = BufWriter::new(
        File::create(output.join("original.jsonl")).map_err(|error| error.to_string())?,
    );
    let mut perturbed_file = BufWriter::new(
        File::create(output.join("perturbed.jsonl")).map_err(|error| error.to_string())?,
    );
    for step in 0..=STEPS {
        if step.is_multiple_of(SNAPSHOT_STEPS) {
            let time = step as f64 * DT;
            write_snapshot(&mut original_file, time, &original)?;
            write_snapshot(&mut perturbed_file, time, &perturbed)?;
        }
        if step == STEPS {
            break;
        }
        original = RungeKutta4.step(&original, DT, |state| grid.rate(state, nu));
        perturbed = RungeKutta4.step(&perturbed, DT, |state| grid.rate(state, nu));
        original = grid.filter(&original);
        perturbed = grid.filter(&perturbed);
    }
    original_file.flush().map_err(|error| error.to_string())?;
    perturbed_file.flush().map_err(|error| error.to_string())?;
    println!("{}\tM={largest_component:.17e}", field.case);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sensitivity-data: {error}");
        process::exit(2);
    }
}
