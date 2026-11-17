mod magnet;
mod nqueens;

use serde_json::{json, Value};
use std::{env, error::Error, fs, time::Instant};

fn and(a: i32, b: i32, z: i32) -> i32 {
    a * b - 2 * a * z - 2 * b * z - a - b + 2 * z
}

fn xor(a: i32, b: i32, z: i32, w: i32) -> i32 {
    a * b - a * z - 2 * a * w - b * z - 2 * b * w + 2 * z * w - a - b + z + 2 * w
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments
        .windows(2)
        .any(|pair| pair[0] == "--mode" && pair[1] != "gadgets")
    {
        return magnet::run(&arguments);
    }
    let start = Instant::now();
    let mut mode = None;
    let mut seed = None;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().ok_or("each option requires a value")?;
        match arg.as_str() {
            "--mode" if mode.is_none() => mode = Some(value),
            "--seed" if seed.is_none() => seed = Some(value.parse::<u64>()?),
            "--out" if out.is_none() => out = Some(value),
            _ => return Err(format!("unknown or repeated option: {arg}").into()),
        }
    }
    let mode = mode.ok_or("missing --mode")?;
    if mode != "gadgets" {
        return Err(format!("unsupported mode: {mode}").into());
    }
    let seed = seed.ok_or("missing --seed")?;
    let out = out.ok_or("missing --out")?;

    let mut evidence: Vec<Value> = Vec::new();
    for (gate, width) in [
        ("AND", 3),
        ("OR", 3),
        ("NOT", 2),
        ("COPY", 2),
        ("SET0", 1),
        ("SET1", 1),
        ("XOR", 3),
        ("circuit", 4),
    ] {
        for state in 0..(1 << width) {
            let spins: Vec<i32> = (0..width)
                .rev()
                .map(|i| 2 * ((state >> i) & 1) - 1)
                .collect();
            let mut helpers = None;
            let energy = match gate {
                "AND" => and(spins[0], spins[1], spins[2]),
                "OR" => {
                    let (a, b, z) = (spins[0], spins[1], spins[2]);
                    a * b - 2 * a * z - 2 * b * z + a + b - 2 * z
                }
                "NOT" => spins[0] * spins[1],
                "COPY" => -spins[0] * spins[1],
                "SET0" => spins[0],
                "SET1" => -spins[0],
                "XOR" => {
                    let values = [
                        xor(spins[0], spins[1], spins[2], -1),
                        xor(spins[0], spins[1], spins[2], 1),
                    ];
                    helpers = Some(values);
                    values[0].min(values[1])
                }
                "circuit" => and(spins[0], spins[1], spins[2]) + spins[2] * spins[3] - spins[3],
                _ => unreachable!(),
            };
            let bits = format!("{state:0width$b}");
            let mut row = json!({"gate": gate, "bits": bits, "energy": energy});
            if let Some(values) = helpers {
                row["helper_energies"] = json!(values);
            }
            evidence.push(row);
        }
    }
    fs::create_dir_all(&out)?;
    let runtime_ms = start.elapsed().as_secs_f64() * 1000.0;
    let run = json!({
        "schema_version": 1, "mode": mode, "seed": seed,
        "parameters": {"out": out},
        "summary": {"row_count": evidence.len(), "runtime_ms": runtime_ms},
        "evidence": evidence,
    });
    fs::write(
        format!("{out}/run.json"),
        serde_json::to_string_pretty(&run)? + "\n",
    )?;
    println!("mode\tmetric\tvalue\truntime_ms");
    for row in run["evidence"].as_array().unwrap() {
        println!(
            "gadgets\t{}/{}\t{}\t{}",
            row["gate"].as_str().unwrap(),
            row["bits"].as_str().unwrap(),
            row["energy"],
            runtime_ms
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("tnsample: {error}");
        std::process::exit(1);
    }
}
