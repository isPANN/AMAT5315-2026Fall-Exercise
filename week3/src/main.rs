use std::{env, fs, io, path::PathBuf, process};

struct Args {
    update: String,
    l: usize,
    t_from: f64,
    t_to: f64,
    t_step: f64,
    discard: u64,
    measure: u64,
    every: u64,
    seed: u64,
    out: PathBuf,
}

fn value<'a>(args: &'a [String], name: &str) -> Result<&'a str, String> {
    let i = args
        .iter()
        .position(|arg| arg == name)
        .ok_or_else(|| format!("missing required argument {name}"))?;
    args.get(i + 1)
        .map(String::as_str)
        .ok_or_else(|| format!("missing value for {name}"))
}

fn parse<T: std::str::FromStr>(args: &[String], name: &str) -> Result<T, String> {
    value(args, name)?
        .parse()
        .map_err(|_| format!("invalid value for {name}"))
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<_> = env::args().skip(1).collect();
    if raw.len() % 2 != 0 {
        return Err("each argument must have a value".into());
    }
    const NAMES: [&str; 10] = [
        "--update",
        "--l",
        "--t-from",
        "--t-to",
        "--t-step",
        "--discard",
        "--measure",
        "--every",
        "--seed",
        "--out",
    ];
    for i in (0..raw.len()).step_by(2) {
        if !NAMES.contains(&raw[i].as_str()) {
            return Err(format!("unknown argument {}", raw[i]));
        }
    }
    let update = value(&raw, "--update")?;
    if update != "metropolis" && update != "wolff" {
        return Err("--update must be metropolis or wolff".into());
    }
    let args = Args {
        update: update.into(),
        l: parse(&raw, "--l")?,
        t_from: parse(&raw, "--t-from")?,
        t_to: parse(&raw, "--t-to")?,
        t_step: parse(&raw, "--t-step")?,
        discard: parse(&raw, "--discard")?,
        measure: parse(&raw, "--measure")?,
        every: raw
            .iter()
            .position(|arg| arg == "--every")
            .map(|_| parse(&raw, "--every"))
            .transpose()?
            .unwrap_or(0),
        seed: parse(&raw, "--seed")?,
        out: value(&raw, "--out")?.into(),
    };
    if args.l < 2 {
        return Err("--l must be at least 2".into());
    }
    args.l.checked_mul(args.l).ok_or("--l is too large")?;
    if !args.t_from.is_finite() || args.t_from <= 0.0 {
        return Err("--t-from must be positive and finite".into());
    }
    if !args.t_to.is_finite() || args.t_to < args.t_from {
        return Err("--t-to must be finite and at least --t-from".into());
    }
    if !args.t_step.is_finite() || args.t_step <= 0.0 {
        return Err("--t-step must be positive and finite".into());
    }
    if args.measure == 0 {
        return Err("--measure must be positive".into());
    }
    Ok(args)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn index(&mut self, n: usize) -> usize {
        let n = n as u64;
        let limit = u64::MAX - u64::MAX % n;
        loop {
            let x = self.next();
            if x < limit {
                return (x % n) as usize;
            }
        }
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }
}

fn neighbors(i: usize, l: usize) -> [usize; 4] {
    let row = i / l;
    let col = i % l;
    [
        row * l + (col + 1) % l,
        row * l + (col + l - 1) % l,
        ((row + 1) % l) * l + col,
        ((row + l - 1) % l) * l + col,
    ]
}

fn metropolis(spins: &mut [i8], l: usize, t: f64, rng: &mut Rng, m: &mut i64, e: &mut i64) -> u64 {
    let mut accepted = 0;
    for _ in 0..spins.len() {
        let i = rng.index(spins.len());
        let neighbor_sum: i64 = neighbors(i, l).map(|j| spins[j] as i64).iter().sum();
        let delta_e = 2 * spins[i] as i64 * neighbor_sum;
        if delta_e <= 0 || rng.unit() < (-delta_e as f64 / t).exp() {
            let old = spins[i] as i64;
            spins[i] = -spins[i];
            *m -= 2 * old;
            *e += delta_e;
            accepted += 1;
        }
    }
    accepted
}

fn wolff(spins: &mut [i8], l: usize, t: f64, rng: &mut Rng, m: &mut i64, e: &mut i64) -> u64 {
    let seed = rng.index(spins.len());
    let cluster_spin = spins[seed];
    let bond_probability = 1.0 - (-2.0 / t).exp();
    let mut included = vec![false; spins.len()];
    let mut cluster = vec![seed];
    included[seed] = true;
    let mut cursor = 0;
    while cursor < cluster.len() {
        for neighbor in neighbors(cluster[cursor], l) {
            if !included[neighbor]
                && spins[neighbor] == cluster_spin
                && rng.unit() < bond_probability
            {
                included[neighbor] = true;
                cluster.push(neighbor);
            }
        }
        cursor += 1;
    }

    let mut delta_e = 0;
    for &i in &cluster {
        for neighbor in neighbors(i, l) {
            if !included[neighbor] {
                delta_e += 2 * spins[i] as i64 * spins[neighbor] as i64;
            }
        }
        spins[i] = -spins[i];
    }
    *m -= 2 * cluster_spin as i64 * cluster.len() as i64;
    *e += delta_e;
    cluster.len() as u64
}

fn temperatures(from: f64, to: f64, step: f64) -> Vec<f64> {
    let steps = (to - from) / step;
    let roundoff = 8.0 * f64::EPSILON * steps.abs().max(1.0);
    (0..=(steps + roundoff).floor() as u64)
        .map(|n| from + n as f64 * step)
        .collect()
}

fn run(args: Args) -> io::Result<()> {
    fs::create_dir_all(&args.out)?;
    let grid = temperatures(args.t_from, args.t_to, args.t_step);
    let is_metropolis = args.update == "metropolis";
    let update = if is_metropolis {
        metropolis as fn(&mut [i8], usize, f64, &mut Rng, &mut i64, &mut i64) -> u64
    } else {
        wolff
    };
    let time_unit = if is_metropolis {
        "sweep"
    } else {
        "cluster_flip"
    };
    fs::write(
        args.out.join("run.json"),
        format!(
            "{{\"L\":{},\"update\":\"{}\",\"t_grid\":{:?},\"discard\":{},\"measure\":{},\"seed\":{},\"sample_every\":1,\"time_unit\":\"{}\"}}\n",
            args.l, args.update, grid, args.discard, args.measure, args.seed, time_unit
        ),
    )?;

    let mut series = String::new();
    let mut frames = String::new();
    let sites = args.l * args.l;
    let mut spins = vec![1_i8; sites];
    let mut magnetization = sites as i64;
    let mut energy = -2 * sites as i64;
    let mut rng = Rng(args.seed);
    let mut total_sweeps = 0_u64;

    println!(
        "T\tmean_abs_M\t{}",
        if is_metropolis {
            "acceptance_rate"
        } else {
            "mean_cluster_size"
        }
    );
    for &t in &grid {
        let mut update_total = 0_u64;
        for _ in 0..args.discard {
            update_total += update(
                &mut spins,
                args.l,
                t,
                &mut rng,
                &mut magnetization,
                &mut energy,
            );
            total_sweeps += 1;
        }
        let mut mean_abs_m = 0.0;
        for measured_sweep in 1..=args.measure {
            let update_size = update(
                &mut spins,
                args.l,
                t,
                &mut rng,
                &mut magnetization,
                &mut energy,
            );
            update_total += update_size;
            total_sweeps += 1;
            let m = magnetization as f64 / sites as f64;
            let e = energy as f64 / sites as f64;
            mean_abs_m += m.abs();
            if is_metropolis {
                series.push_str(&format!(
                    "{{\"L\":{},\"T\":{:.6},\"sweep\":{},\"M\":{:.6},\"E\":{:.6}}}\n",
                    args.l, t, measured_sweep, m, e
                ));
            } else {
                series.push_str(&format!(
                    "{{\"L\":{},\"T\":{:.6},\"sweep\":{},\"M\":{:.6},\"E\":{:.6},\"cluster_size\":{}}}\n",
                    args.l, t, measured_sweep, m, e, update_size
                ));
            }
            if args.every > 0 && measured_sweep % args.every == 0 {
                frames.push_str(&format!(
                    "{{\"L\":{},\"T\":{:.6},\"sweep\":{},\"m\":{:.6},\"spins\":{:?}}}\n",
                    args.l, t, total_sweeps, m, spins
                ));
            }
        }
        let steps = (args.discard + args.measure) as f64;
        let update_statistic = if is_metropolis {
            update_total as f64 / (steps * sites as f64)
        } else {
            update_total as f64 / steps
        };
        println!(
            "{:.6}\t{:.6}\t{:.6}",
            t,
            mean_abs_m / args.measure as f64,
            update_statistic
        );
    }
    fs::write(args.out.join("series.jsonl"), series)?;
    fs::write(args.out.join("spins.jsonl"), frames)
}

fn main() {
    let args = parse_args().unwrap_or_else(|message| {
        eprintln!("error: {message}");
        process::exit(2);
    });
    if let Err(error) = run(args) {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_keep_incremental_observables_consistent() {
        let mut rng = Rng(2026);
        for l in [2, 5] {
            let mut spins = vec![1; l * l];
            let mut m = (l * l) as i64;
            let mut e = -2 * m;
            for _ in 0..20 {
                metropolis(&mut spins, l, 2.3, &mut rng, &mut m, &mut e);
                let cluster_size = wolff(&mut spins, l, 2.3, &mut rng, &mut m, &mut e);
                assert!((1..=l as u64 * l as u64).contains(&cluster_size));
            }
            let direct_m: i64 = spins.iter().map(|&s| s as i64).sum();
            let direct_e: i64 = (0..l * l)
                .map(|i| {
                    -(spins[i] as i64)
                        * (spins[neighbors(i, l)[0]] as i64 + spins[neighbors(i, l)[2]] as i64)
                })
                .sum();
            assert_eq!((m, e), (direct_m, direct_e));
        }
    }

    #[test]
    fn temperature_grid_includes_reached_decimal_endpoint() {
        assert_eq!(temperatures(0.1, 0.3, 0.1).len(), 3);
        assert_eq!(temperatures(0.1, 0.29, 0.1).len(), 2);
    }
}
