use std::{fs, process::Command};

#[test]
fn cli_writes_samples_and_reports_failed_or_malformed_input() {
    let bin = env!("CARGO_BIN_EXE_md");
    let dir = std::env::temp_dir().join(format!("md-cli-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let run = Command::new(bin)
        .args([
            "run",
            "--eq-steps",
            "0",
            "--steps",
            "1000",
            "--sample-every",
            "10",
            "--out",
        ])
        .arg(&dir)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let (config, mut frames) = md::trajectory::read_run(&dir).unwrap();
    assert_eq!(frames.len(), 100);
    assert_eq!(frames[0].step, 10);
    assert_eq!(frames[99].step, 1000);
    assert_eq!(config.n, 100);
    assert_eq!(config.temperature, 0.5);
    assert_eq!(config.seed, 2026);
    // Equal speeds intentionally fail the speed-shape gate at a well-defined temperature.
    for frame in &mut frames {
        for velocity in &mut frame.vel {
            *velocity = [1.0, 0.0];
        }
    }
    let text = frames
        .iter()
        .map(|f| serde_json::to_string(f).unwrap() + "\n")
        .collect::<String>();
    fs::write(dir.join("traj.jsonl"), text).unwrap();
    let failed = Command::new(bin).arg("check").arg(&dir).output().unwrap();
    assert!(!failed.status.success());
    let report = String::from_utf8(failed.stdout).unwrap();
    for label in ["Secular drift", "Temperature", "Speed shape", "FAIL"] {
        assert!(report.contains(label));
    }
    fs::write(dir.join("traj.jsonl"), "{broken}\n").unwrap();
    assert!(
        !Command::new(bin)
            .arg("check")
            .arg(&dir)
            .output()
            .unwrap()
            .status
            .success()
    );
    assert!(
        !Command::new(bin)
            .args(["run", "--unknown"])
            .output()
            .unwrap()
            .status
            .success()
    );
    fs::remove_dir_all(&dir).unwrap();
}
