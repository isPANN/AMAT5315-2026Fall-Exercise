use serde::Serialize;
use std::{
    collections::HashMap,
    env,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::PathBuf,
    process,
};
use time_stepping::{
    ExplicitMidpoint, ForwardEuler, Integrator, RungeKutta4,
    vorticity::{SpectralGrid, VelocityField},
};

struct Args {
    method: String,
    nu: f64,
    dt: f64,
    t_end: f64,
    snapshot_steps: u64,
    snapshot_every: f64,
    out: PathBuf,
}

#[derive(Serialize)]
struct Run<'a> {
    case: &'a str,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[i32; 2]>,
    method: &'a str,
    nu: f64,
    dt: f64,
    t_end: f64,
    snapshot_every: f64,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<_> = env::args().skip(1).collect();
    if !raw.len().is_multiple_of(2) {
        return Err("each option must have a value".into());
    }
    let allowed = ["--method", "--nu", "--dt", "--t-end", "--every", "--out"];
    let mut options = HashMap::new();
    for pair in raw.as_chunks::<2>().0 {
        if !allowed.contains(&pair[0].as_str()) {
            return Err(format!("unknown option {}", pair[0]));
        }
        if options.insert(pair[0].clone(), pair[1].clone()).is_some() {
            return Err(format!("duplicate option {}", pair[0]));
        }
    }
    let value = |name: &str| {
        options
            .get(name)
            .cloned()
            .ok_or_else(|| format!("missing required option {name}"))
    };
    let number = |name: &str| {
        value(name)?
            .parse::<f64>()
            .map_err(|_| format!("invalid value for {name}"))
    };
    let method = value("--method")?;
    if !["euler", "rk2", "rk4"].contains(&method.as_str()) {
        return Err("--method must be euler, rk2, or rk4".into());
    }
    let nu = number("--nu")?;
    let dt = number("--dt")?;
    let t_end = number("--t-end")?;
    let every = number("--every")?;
    if !nu.is_finite() || nu < 0.0 {
        return Err("--nu must be nonnegative and finite".into());
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err("--dt must be positive and finite".into());
    }
    if !t_end.is_finite() || t_end < 0.0 {
        return Err("--t-end must be nonnegative and finite".into());
    }
    if !every.is_finite() || every <= 0.0 {
        return Err("--every must be positive and finite".into());
    }
    let snapshot_ratio = every / dt;
    if !snapshot_ratio.is_finite() {
        return Err("--every divided by --dt is too large".into());
    }
    let snapshot_steps = snapshot_ratio.round() as u64;
    if snapshot_steps == 0 {
        return Err("--every rounds to zero time steps".into());
    }
    Ok(Args {
        method,
        nu,
        dt,
        t_end,
        snapshot_steps,
        snapshot_every: snapshot_steps as f64 * dt,
        out: value("--out")?.into(),
    })
}

fn advance<F>(method: &str, state: &[f64], step: f64, rate: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    match method {
        "euler" => ForwardEuler.step(state, step, rate),
        "rk2" => ExplicitMidpoint.step(state, step, rate),
        "rk4" => RungeKutta4.step(state, step, rate),
        _ => unreachable!(),
    }
}

fn write_array(writer: &mut impl Write, values: &[f64]) -> io::Result<()> {
    writer.write_all(b"[")?;
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            writer.write_all(b",")?;
        }
        write!(writer, "{value:.6}")?;
    }
    writer.write_all(b"]")
}

fn write_frame(
    writer: &mut impl Write,
    time: f64,
    step: u64,
    u: &[f64],
    v: &[f64],
    omega: &[f64],
) -> io::Result<()> {
    write!(writer, "{{\"t\":{time:.6},\"step\":{step},\"u\":")?;
    write_array(writer, u)?;
    writer.write_all(b",\"v\":")?;
    write_array(writer, v)?;
    writer.write_all(b",\"omega\":")?;
    write_array(writer, omega)?;
    writer.write_all(b"}\n")
}

fn run() -> Result<bool, String> {
    let args = parse_args()?;
    let field: VelocityField =
        serde_json::from_reader(io::stdin().lock()).map_err(|error| error.to_string())?;
    let size = field.n.checked_mul(field.n).ok_or("n is too large")?;
    if field.n < 3 {
        return Err("n must be at least 3".into());
    }
    if field.u.len() != size || field.v.len() != size {
        return Err("u and v must each contain n*n values".into());
    }
    let grid = SpectralGrid::new(field.n);
    let mut omega = grid.vorticity_from_velocity(&field.u, &field.v);

    fs::create_dir_all(&args.out).map_err(|error| error.to_string())?;
    let run = Run {
        case: &field.case,
        n: field.n,
        seed: field.seed,
        k_band: field.k_band,
        method: &args.method,
        nu: args.nu,
        dt: args.dt,
        t_end: args.t_end,
        snapshot_every: args.snapshot_every,
    };
    let run_file = File::create(args.out.join("run.json")).map_err(|error| error.to_string())?;
    serde_json::to_writer(run_file, &run).map_err(|error| error.to_string())?;
    let fields_file =
        File::create(args.out.join("fields.jsonl")).map_err(|error| error.to_string())?;
    let mut fields = BufWriter::new(fields_file);

    println!("t\tE\tZ");
    let mut time = 0.0;
    let mut step = 0_u64;
    loop {
        let (u, v) = grid.velocity_from_vorticity(&omega);
        let energy = u
            .iter()
            .zip(v.iter())
            .map(|(&u, &v)| 0.5 * (u * u + v * v))
            .sum::<f64>()
            / size as f64;
        let enstrophy = omega.iter().map(|omega| 0.5 * omega * omega).sum::<f64>() / size as f64;
        let on_step_grid =
            (time - step as f64 * args.dt).abs() <= 16.0 * f64::EPSILON * time.abs().max(1.0);
        let snapshot = step.is_multiple_of(args.snapshot_steps) && on_step_grid;
        if snapshot || !energy.is_finite() {
            println!("{time:.6}\t{energy:.12e}\t{enstrophy:.12e}");
        }
        if !energy.is_finite() {
            fields.flush().map_err(|error| error.to_string())?;
            return Ok(false);
        }
        if snapshot {
            write_frame(&mut fields, time, step, &u, &v, &omega)
                .map_err(|error| error.to_string())?;
        }
        if time >= args.t_end {
            break;
        }
        let step_size = args.dt.min(args.t_end - time);
        omega = advance(&args.method, &omega, step_size, |state| {
            grid.rate(state, args.nu)
        });
        omega = grid.filter(&omega);
        time += step_size;
        step += 1;
    }
    fields.flush().map_err(|error| error.to_string())?;
    Ok(true)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => process::exit(1),
        Err(error) => {
            eprintln!("fluid: {error}");
            process::exit(2);
        }
    }
}
