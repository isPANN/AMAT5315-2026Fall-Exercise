use clap::{Args, Parser, Subcommand};
use md::trajectory::{RunConfig, geometry, read_run, write_run};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "md",
    about = "Lennard-Jones fluid simulation and saved-trajectory analysis"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}
#[derive(Subcommand)]
enum Commands {
    Run(RunArgs),
    Check {
        directory: PathBuf,
    },
    Video {
        directory: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}
#[derive(Args)]
struct RunArgs {
    #[arg(long, default_value_t = 100)]
    n: usize,
    #[arg(long, default_value_t = 0.8)]
    rho: f64,
    #[arg(long, default_value_t = 0.5)]
    temperature: f64,
    #[arg(long, default_value_t = 0.01)]
    dt: f64,
    #[arg(long, default_value_t = 2000)]
    eq_steps: usize,
    #[arg(long, default_value_t = 10000)]
    steps: usize,
    #[arg(long, default_value_t = 50)]
    sample_every: usize,
    #[arg(long, default_value_t = 2026)]
    seed: u64,
    #[arg(long)]
    out: PathBuf,
}
fn main() -> md::Result<()> {
    match Cli::parse().command {
        Commands::Run(args) => {
            let config = RunConfig {
                n: args.n,
                rho: args.rho,
                box_size: geometry(args.n, args.rho)?,
                dt: args.dt,
                temperature: args.temperature,
                eq_steps: args.eq_steps,
                steps: args.steps,
                sample_every: args.sample_every,
                seed: args.seed,
                integrator: "velocity-verlet".into(),
            };
            write_run(&config, &args.out)?;
            println!(
                "Saved {} frames to {}",
                config.steps / config.sample_every,
                args.out.display()
            );
        }
        Commands::Check { directory } => {
            let (config, frames) = read_run(&directory)?;
            let report = md::analysis::check(&config, &frames)?;
            println!(
                "T_speed = {:.8}; target = {:.8}",
                report.temperature, config.temperature
            );
            for (name, value, limit) in [
                ("Secular drift", report.drift, 0.002),
                (
                    "Temperature",
                    (report.temperature - config.temperature).abs(),
                    0.05,
                ),
                ("Speed shape", report.speed_shape, 2.0),
            ] {
                println!(
                    "{name}: {value:.8e} < {limit:.8e} {}",
                    if value < limit { "PASS" } else { "FAIL" }
                );
            }
            if !report.passed(config.temperature) {
                return Err("physics checks failed".into());
            }
        }
        Commands::Video { directory, out } => {
            let (config, frames) = read_run(&directory)?;
            md::video::record(&config, &frames, &out)?;
            println!("Saved {} frames to {}", frames.len(), out.display());
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn abbreviated_run_matches_explicit_contract_flags() {
        let Commands::Run(a) = Cli::try_parse_from(["md", "run", "--out", "artifacts"])
            .unwrap()
            .command
        else {
            panic!()
        };
        let Commands::Run(b) = Cli::try_parse_from([
            "md",
            "run",
            "--n",
            "100",
            "--rho",
            "0.8",
            "--temperature",
            "0.5",
            "--dt",
            "0.01",
            "--eq-steps",
            "2000",
            "--steps",
            "10000",
            "--sample-every",
            "50",
            "--seed",
            "2026",
            "--out",
            "artifacts",
        ])
        .unwrap()
        .command
        else {
            panic!()
        };
        assert_eq!(
            (
                a.n,
                a.rho,
                a.temperature,
                a.dt,
                a.eq_steps,
                a.steps,
                a.sample_every,
                a.seed,
                a.out
            ),
            (
                b.n,
                b.rho,
                b.temperature,
                b.dt,
                b.eq_steps,
                b.steps,
                b.sample_every,
                b.seed,
                b.out
            )
        );
    }
}
