use serde::Deserialize;
use serde_json::Value;
use std::{collections::BTreeMap, fs, path::PathBuf, process::Command};

fn run(mode: &str, n: usize, extra: &[&str]) -> Value {
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("queens-{mode}-{n}-{}", std::process::id()));
    let result = Command::new(env!("CARGO_BIN_EXE_tnsample"))
        .args([
            "--mode",
            mode,
            "--seed",
            "2026",
            "--out",
            out.to_str().unwrap(),
            "--n",
            &n.to_string(),
            "--order",
            if n >= 5 { "treesa" } else { "greedy" },
        ])
        .args(extra)
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
    let bytes = fs::read(out.join("run.json")).unwrap();
    let mut deserializer = serde_json::Deserializer::from_slice(&bytes);
    deserializer.disable_recursion_limit();
    Value::deserialize(&mut deserializer).unwrap()
}

#[test]
fn counts_known_boards_and_supports_planning() {
    for (n, count) in [(1, "1"), (2, "0"), (3, "0"), (4, "2"), (5, "10")] {
        let result = run("count", n, &[]);
        assert_eq!(result["summary"]["count"], count);
    }
    let plan = run("plan", 4, &[]);
    assert!(plan["summary"]["tc"].as_f64().unwrap().is_finite());
    assert!(plan["summary"]["sc"].as_f64().unwrap().is_finite());
}

#[test]
fn sampled_boards_are_valid_and_uniform() {
    for (n, expected, tolerance) in [(4, 2000isize, 200isize), (5, 400, 100)] {
        let result = run("sample", n, &["--samples", "4000"]);
        let samples = result["evidence"]["samples"].as_array().unwrap();
        assert_eq!(samples.len(), 4000);
        let mut counts = BTreeMap::new();
        for sample in samples {
            let board = sample["board"].as_array().unwrap();
            assert_eq!(board.len(), n);
            let cols: Vec<usize> = board
                .iter()
                .map(|row| {
                    let row = row.as_array().unwrap();
                    assert_eq!(row.len(), n);
                    assert_eq!(row.iter().filter(|x| x.as_u64().unwrap() == 1).count(), 1);
                    assert!(row.iter().all(|x| x.as_u64().unwrap() <= 1));
                    row.iter().position(|x| x.as_u64().unwrap() == 1).unwrap()
                })
                .collect();
            for i in 0..n {
                for j in 0..i {
                    assert_ne!(cols[i], cols[j]);
                    assert_ne!(cols[i].abs_diff(cols[j]), i - j);
                }
            }
            *counts.entry(cols).or_insert(0usize) += 1;
        }
        assert_eq!(counts.len(), 4000 / expected as usize);
        for count in counts.values() {
            assert!((*count as isize - expected).abs() < tolerance);
        }
    }
}

#[test]
fn sampling_an_unsatisfiable_board_fails() {
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/queens-no-solution");
    let result = Command::new(env!("CARGO_BIN_EXE_tnsample"))
        .args([
            "--mode",
            "sample",
            "--seed",
            "1",
            "--out",
            out.to_str().unwrap(),
            "--n",
            "3",
            "--order",
            "greedy",
            "--samples",
            "1",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!result.stderr.is_empty());
}

#[test]
fn rejects_magnet_parameters_for_queens() {
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/queens-conflict");
    let result = Command::new(env!("CARGO_BIN_EXE_tnsample"))
        .args([
            "--mode",
            "count",
            "--seed",
            "1",
            "--out",
            out.to_str().unwrap(),
            "--n",
            "4",
            "--beta",
            "1",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!result.stderr.is_empty());
}

#[test]
fn eight_queens_count_and_samples_match_backtracking() {
    fn enumerate(cols: &mut Vec<usize>, solutions: &mut Vec<Vec<usize>>) {
        let row = cols.len();
        if row == 8 {
            solutions.push(cols.clone());
            return;
        }
        for col in 0..8 {
            if cols
                .iter()
                .enumerate()
                .all(|(r, &c)| c != col && c.abs_diff(col) != row - r)
            {
                cols.push(col);
                enumerate(cols, solutions);
                cols.pop();
            }
        }
    }
    let mut solutions = Vec::new();
    enumerate(&mut Vec::new(), &mut solutions);
    assert_eq!(solutions.len(), 92);
    let result = run("sample", 8, &["--samples", "200"]);
    assert_eq!(result["summary"]["count"], "92");
    let samples = result["evidence"]["samples"].as_array().unwrap();
    assert_eq!(samples.len(), 200);
    for sample in samples {
        let board = sample["board"].as_array().unwrap();
        assert_eq!(board.len(), 8);
        let columns: Vec<_> = board
            .iter()
            .map(|row| {
                let row = row.as_array().unwrap();
                assert_eq!(row.len(), 8);
                assert_eq!(row.iter().filter(|v| v.as_u64().unwrap() == 1).count(), 1);
                assert!(row.iter().all(|v| v.as_u64().unwrap() <= 1));
                row.iter().position(|v| v.as_u64().unwrap() == 1).unwrap()
            })
            .collect();
        assert!(solutions.contains(&columns));
    }
}
