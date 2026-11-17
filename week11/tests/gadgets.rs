use serde_json::Value;
use std::{fs, process::Command};

#[test]
fn enumerates_gate_and_circuit_energies_in_stdout_and_run_evidence() {
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("tnsample-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&out);
    let output = Command::new(env!("CARGO_BIN_EXE_tnsample"))
        .args(["--mode", "gadgets", "--seed", "2026", "--out"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let run: Value = serde_json::from_slice(&fs::read(out.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["schema_version"], 1);
    assert_eq!(run["mode"], "gadgets");
    assert_eq!(run["seed"], 2026);
    assert_eq!(run["parameters"]["out"], out.to_str().unwrap());
    let cases: [(&str, usize, &[i64]); 8] = [
        ("AND", 3, &[-3, 9, -3, 1, -3, 1, 1, -3]),
        ("OR", 3, &[-3, 1, 1, -3, 1, -3, 9, -3]),
        ("NOT", 2, &[1, -1, -1, 1]),
        ("COPY", 2, &[-1, 1, 1, -1]),
        ("SET0", 1, &[-1, 1]),
        ("SET1", 1, &[1, -1]),
        ("XOR", 3, &[-4, -2, -2, -4, -2, -4, -4, -2]),
        (
            "circuit",
            4,
            &[-1, -5, 9, 9, -1, -5, 1, 1, -1, -5, 1, 1, 3, -1, -3, -3],
        ),
    ];
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    assert_eq!(lines.next(), Some("mode\tmetric\tvalue\truntime_ms"));
    let rows = run["evidence"].as_array().unwrap();
    assert_eq!(run["summary"]["row_count"], 52);
    assert_eq!(rows.len(), 52);
    let mut index = 0;
    for (gate, width, energies) in cases {
        for (bits, energy) in energies.iter().enumerate() {
            let bits = format!("{bits:0width$b}");
            let row = &rows[index];
            assert_eq!(row["gate"], gate);
            assert_eq!(row["bits"], bits);
            assert_eq!(row["energy"], *energy);
            if gate == "XOR" {
                let expected_helpers = [
                    [-4, 4],
                    [-2, 14],
                    [-2, -2],
                    [-4, 4],
                    [-2, -2],
                    [-4, 4],
                    [4, -4],
                    [-2, -2],
                ];
                assert_eq!(
                    row["helper_energies"],
                    serde_json::json!(expected_helpers[usize::from_str_radix(&bits, 2).unwrap()])
                );
                let helpers = row["helper_energies"].as_array().unwrap();
                assert_eq!(helpers.len(), 2);
                assert_eq!(
                    helpers.iter().map(|v| v.as_i64().unwrap()).min().unwrap(),
                    *energy
                );
            }
            let fields: Vec<_> = lines.next().unwrap().split('\t').collect();
            assert_eq!(fields.len(), 4);
            assert_eq!(fields[0], "gadgets");
            assert_eq!(fields[1], format!("{gate}/{bits}"));
            assert_eq!(fields[2].parse::<i64>().unwrap(), *energy);
            assert!(fields[3].parse::<f64>().unwrap() >= 0.0);
            index += 1;
        }
    }
    assert!(lines.next().is_none());
    fs::remove_dir_all(out).unwrap();
}

#[test]
fn requires_all_arguments_and_rejects_other_modes() {
    for args in [
        &["--mode", "gadgets"][..],
        &["--mode", "other", "--seed", "1", "--out", "unused"][..],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_tnsample"))
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(!output.stderr.is_empty());
    }
}
