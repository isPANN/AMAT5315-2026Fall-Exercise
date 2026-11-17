use serde_json::{json, Value};
use std::{fs, path::PathBuf, process::Command};

fn dir(name: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("magnet-{name}-{}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
fn run(args: &[&str]) -> Value {
    let result = Command::new(env!("CARGO_BIN_EXE_tnsample"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8(result.stdout)
        .unwrap()
        .starts_with("mode\tmetric\tvalue\truntime_ms\n"));
    let out = args[args.iter().position(|s| *s == "--out").unwrap() + 1];
    serde_json::from_slice(&fs::read(PathBuf::from(out).join("run.json")).unwrap()).unwrap()
}

#[test]
fn exact_lattice_and_bundle_contractions() {
    let path = dir("count");
    let out = path.to_str().unwrap();
    let plan = run(&[
        "--mode", "plan", "--seed", "9", "--out", out, "--l", "2", "--p-af", "0", "--order",
        "greedy",
    ]);
    assert_eq!(plan["summary"]["variables"], 4);
    assert!(plan["summary"]["sc"].as_f64().unwrap() >= 0.0);
    assert!(plan["summary"]["tc"].as_f64().unwrap().is_finite());
    let count = run(&[
        "--mode", "count", "--seed", "9", "--out", out, "--l", "2", "--p-af", "0", "--beta", "0.7",
        "--order", "greedy",
    ]);
    let expected = (2.0 * (2.8_f64).exp() + 12.0 + 2.0 * (-2.8_f64).exp()).ln();
    assert!((count["summary"]["log_z"].as_f64().unwrap() - expected).abs() < 1e-12);
    assert_eq!(count["summary"]["e_min"], -4);
    assert_eq!(count["summary"]["g_0"], "2");
    let bundle = path.join("bundle.json");
    fs::write(&bundle, serde_json::to_vec(&json!({"target":{"type":"SpinGlass","data":{"graph":{"num_vertices":2,"edges":[[0,1]]},"couplings":[2],"fields":[1,0]}}})).unwrap()).unwrap();
    let result = run(&[
        "--mode",
        "count",
        "--seed",
        "9",
        "--out",
        out,
        "--input",
        bundle.to_str().unwrap(),
        "--beta",
        "0.5",
        "--order",
        "greedy",
    ]);
    let expected = ((1.5_f64).exp() + (0.5_f64).exp() + (-0.5_f64).exp() + (-1.5_f64).exp()).ln();
    assert!((result["summary"]["log_z"].as_f64().unwrap() - expected).abs() < 1e-12);
    assert_eq!(result["summary"]["e_min"], -3);
    assert_eq!(result["summary"]["g_0"], "1");
}

#[test]
fn samples_boltzmann_distribution_and_reports_energies() {
    let path = dir("sample");
    let out = path.to_str().unwrap();
    let result = run(&[
        "--mode",
        "sample",
        "--seed",
        "43",
        "--out",
        out,
        "--l",
        "2",
        "--p-af",
        "0",
        "--beta",
        "0.7",
        "--order",
        "greedy",
        "--samples",
        "12000",
    ]);
    let samples = result["evidence"]["samples"].as_array().unwrap();
    assert_eq!(samples.len(), 12000);
    let mut ground = 0;
    for sample in samples {
        let spins: Vec<i64> = sample["spins"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_i64().unwrap())
            .collect();
        assert_eq!(spins.len(), 4);
        assert!(spins.iter().all(|s| s.abs() == 1));
        let energy =
            -spins[0] * spins[1] - spins[0] * spins[2] - spins[1] * spins[3] - spins[2] * spins[3];
        assert_eq!(sample["energy"], energy);
        if energy == -4 {
            ground += 1;
        }
    }
    let expected = 2.0 * (2.8_f64).exp() / (2.0 * (2.8_f64).exp() + 12.0 + 2.0 * (-2.8_f64).exp());
    assert!((ground as f64 / 12000.0 - expected).abs() < 0.02);
}

#[test]
fn rejects_missing_or_conflicting_inputs() {
    let path = dir("invalid");
    for args in [
        vec![
            "--mode",
            "count",
            "--seed",
            "1",
            "--out",
            path.to_str().unwrap(),
            "--l",
            "2",
            "--p-af",
            "0",
        ],
        vec![
            "--mode",
            "plan",
            "--seed",
            "1",
            "--out",
            path.to_str().unwrap(),
            "--l",
            "2",
        ],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_tnsample"))
            .args(args)
            .output()
            .unwrap();
        assert!(!result.status.success());
    }
}

#[test]
fn bundle_samples_follow_fields_and_couplings() {
    let path = dir("bundle-sample");
    let bundle = path.join("bundle.json");
    fs::write(&bundle, serde_json::to_vec(&json!({"target":{"type":"SpinGlass","data":{"graph":{"num_vertices":2,"edges":[[0,1]]},"couplings":[2],"fields":[1,0]}}})).unwrap()).unwrap();
    let result = run(&[
        "--mode",
        "sample",
        "--seed",
        "73",
        "--out",
        path.to_str().unwrap(),
        "--input",
        bundle.to_str().unwrap(),
        "--beta",
        "0.5",
        "--order",
        "greedy",
        "--samples",
        "12000",
    ]);
    let mut counts = [0usize; 4];
    for sample in result["evidence"]["samples"].as_array().unwrap() {
        let a = sample["spins"][0].as_i64().unwrap();
        let b = sample["spins"][1].as_i64().unwrap();
        assert_eq!(sample["energy"], 2 * a * b + a);
        counts[((a + 1) as usize) + ((b + 1) as usize) / 2] += 1;
    }
    let weights: Vec<f64> = [1.0_f64, -3.0, -1.0, 3.0]
        .iter()
        .map(|e| (-0.5 * e).exp())
        .collect();
    let z: f64 = weights.iter().sum();
    for (count, weight) in counts.iter().zip(weights) {
        assert!((*count as f64 / 12000.0 - weight / z).abs() < 0.02);
    }
}

#[test]
fn handles_low_temperature_and_single_spin_with_default_planner() {
    let path = dir("low-temperature");
    let out = path.to_str().unwrap();
    let single = run(&[
        "--mode", "count", "--seed", "1", "--out", out, "--l", "1", "--p-af", "0", "--beta", "1000",
    ]);
    assert!((single["summary"]["log_z"].as_f64().unwrap() - 2f64.ln()).abs() < 1e-12);
    assert_eq!(single["summary"]["g_0"], "2");
    let square = run(&[
        "--mode", "count", "--seed", "1", "--out", out, "--l", "2", "--p-af", "0", "--beta",
        "1000", "--order", "greedy",
    ]);
    assert!((square["summary"]["log_z"].as_f64().unwrap() - (4000.0 + 2f64.ln())).abs() < 1e-10);
}

#[test]
fn random_lattice_matches_enumeration_and_seed_fixes_bonds() {
    let path = dir("random-lattice");
    let args = [
        "--mode",
        "count",
        "--seed",
        "12",
        "--out",
        path.to_str().unwrap(),
        "--l",
        "2",
        "--p-af",
        "0.5",
        "--beta",
        "0.8",
        "--order",
        "greedy",
    ];
    let first = run(&args);
    let second = run(&args);
    assert_eq!(first["evidence"]["edges"], second["evidence"]["edges"]);
    let edges = first["evidence"]["edges"].as_array().unwrap();
    let mut energies = Vec::new();
    for state in 0..16 {
        let spins: Vec<i64> = (0..4)
            .map(|i| if state & (1 << i) == 0 { -1 } else { 1 })
            .collect();
        energies.push(
            edges
                .iter()
                .map(|e| {
                    e[2].as_i64().unwrap()
                        * spins[e[0].as_u64().unwrap() as usize]
                        * spins[e[1].as_u64().unwrap() as usize]
                })
                .sum::<i64>(),
        );
    }
    let minimum = *energies.iter().min().unwrap();
    let log_z = energies
        .iter()
        .map(|e| (-0.8 * *e as f64).exp())
        .sum::<f64>()
        .ln();
    assert_eq!(first["summary"]["e_min"], minimum);
    assert_eq!(
        first["summary"]["g_0"],
        energies
            .iter()
            .filter(|e| **e == minimum)
            .count()
            .to_string()
    );
    assert!((first["summary"]["log_z"].as_f64().unwrap() - log_z).abs() < 1e-12);
}
