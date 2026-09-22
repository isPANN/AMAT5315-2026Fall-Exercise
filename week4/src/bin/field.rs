use serde_json::to_writer;
use std::{collections::HashMap, env, f64::consts::TAU, io, process};
use time_stepping::vorticity::{SpectralGrid, VelocityField};

fn options(args: &[String], allowed: &[&str]) -> Result<HashMap<String, String>, String> {
    if !args.len().is_multiple_of(2) {
        return Err("each option must have a value".into());
    }
    let mut options = HashMap::new();
    for pair in args.as_chunks::<2>().0 {
        if !allowed.contains(&pair[0].as_str()) {
            return Err(format!("unknown option {}", pair[0]));
        }
        if options.insert(pair[0].clone(), pair[1].clone()).is_some() {
            return Err(format!("duplicate option {}", pair[0]));
        }
    }
    Ok(options)
}

fn parse<T: std::str::FromStr>(options: &HashMap<String, String>, name: &str) -> Result<T, String> {
    options
        .get(name)
        .ok_or_else(|| format!("missing required option {name}"))?
        .parse()
        .map_err(|_| format!("invalid value for {name}"))
}

fn unit(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut value = *seed;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((value ^ (value >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
}

fn taylor_green(args: &[String]) -> Result<VelocityField, String> {
    let options = options(args, &["--n", "--nu", "--t"])?;
    let n: usize = parse(&options, "--n")?;
    if n < 3 {
        return Err("--n must be at least 3".into());
    }
    let size = n.checked_mul(n).ok_or("--n is too large")?;
    let t = options
        .get("--t")
        .map(|value| value.parse().map_err(|_| "invalid value for --t"))
        .transpose()?
        .unwrap_or(0.0_f64);
    if !t.is_finite() || t < 0.0 {
        return Err("--t must be nonnegative and finite".into());
    }
    let nu = if t > 0.0 {
        parse(&options, "--nu")?
    } else {
        options
            .get("--nu")
            .map(|value| value.parse().map_err(|_| "invalid value for --nu"))
            .transpose()?
            .unwrap_or(0.0_f64)
    };
    if !nu.is_finite() || nu < 0.0 {
        return Err("--nu must be nonnegative and finite".into());
    }
    let decay = (-2.0 * nu * t).exp();
    let mut u = Vec::with_capacity(size);
    let mut v = Vec::with_capacity(size);
    for y in 0..n {
        let y = TAU * y as f64 / n as f64;
        for x in 0..n {
            let x = TAU * x as f64 / n as f64;
            u.push(x.cos() * y.sin() * decay);
            v.push(-x.sin() * y.cos() * decay);
        }
    }
    Ok(VelocityField {
        case: "taylor-green".into(),
        n,
        seed: None,
        k_band: None,
        u,
        v,
    })
}

fn random(args: &[String]) -> Result<VelocityField, String> {
    let options = options(args, &["--n", "--seed", "--k-min", "--k-max"])?;
    let n: usize = parse(&options, "--n")?;
    let seed: u64 = parse(&options, "--seed")?;
    let k_min: i32 = parse(&options, "--k-min")?;
    let k_max: i32 = parse(&options, "--k-max")?;
    if n < 3 {
        return Err("--n must be at least 3".into());
    }
    let size = n.checked_mul(n).ok_or("--n is too large")?;
    if k_min < 1 || k_max < k_min {
        return Err("require 1 <= --k-min <= --k-max".into());
    }
    if k_max > (n / 3) as i32 {
        return Err("--k-max exceeds the two-thirds cutoff floor(n/3)".into());
    }

    let mut rng = seed;
    let mut modes = Vec::new();
    for ky in -k_max..=k_max {
        for kx in -k_max..=k_max {
            let radius_squared = kx * kx + ky * ky;
            if (kx > 0 || (kx == 0 && ky > 0))
                && radius_squared >= k_min * k_min
                && radius_squared <= k_max * k_max
            {
                modes.push((kx, ky, TAU * unit(&mut rng)));
            }
        }
    }
    if modes.is_empty() {
        return Err("the requested k band contains no lattice modes".into());
    }

    let mut omega = Vec::with_capacity(size);
    for y in 0..n {
        let y = TAU * y as f64 / n as f64;
        for x in 0..n {
            let x = TAU * x as f64 / n as f64;
            omega.push(
                modes
                    .iter()
                    .map(|&(kx, ky, phase)| (kx as f64 * x + ky as f64 * y + phase).cos())
                    .sum(),
            );
        }
    }
    let grid = SpectralGrid::new(n);
    let (mut u, mut v) = grid.velocity_from_vorticity(&omega);
    let energy = u
        .iter()
        .zip(v.iter())
        .map(|(&u, &v)| 0.5 * (u * u + v * v))
        .sum::<f64>()
        / size as f64;
    let scale = (0.5 / energy).sqrt();
    for value in u.iter_mut().chain(v.iter_mut()) {
        *value *= scale;
    }

    Ok(VelocityField {
        case: "random".into(),
        n,
        seed: Some(seed),
        k_band: Some([k_min, k_max]),
        u,
        v,
    })
}

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    let (case, args) = args
        .split_first()
        .ok_or_else(|| "usage: field <taylor-green|random> [options]".to_string())?;
    let field = match case.as_str() {
        "taylor-green" => taylor_green(args)?,
        "random" => random(args)?,
        _ => return Err(format!("unknown case {case}")),
    };
    to_writer(io::stdout().lock(), &field).map_err(|error| error.to_string())?;
    println!();
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("field: {error}");
        process::exit(2);
    }
}
