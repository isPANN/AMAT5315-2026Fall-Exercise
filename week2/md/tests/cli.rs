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
    assert_eq!(config.force, md::Force::Cells);
    assert_eq!(config.n, 100);
    assert_eq!(config.temperature, 0.5);
    assert_eq!(config.ramp_to, None);
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
    let naive = Command::new(bin)
        .args([
            "run",
            "--force",
            "naive",
            "--eq-steps",
            "0",
            "--steps",
            "10",
            "--sample-every",
            "10",
            "--out",
        ])
        .arg(&dir)
        .output()
        .unwrap();
    assert!(
        naive.status.success(),
        "{}",
        String::from_utf8_lossy(&naive.stderr)
    );
    let (config, _) = md::trajectory::read_run(&dir).unwrap();
    assert_eq!(config.force, md::Force::Naive);
    assert!(
        !Command::new(bin)
            .args(["run", "--force", "unknown", "--out"])
            .arg(&dir)
            .output()
            .unwrap()
            .status
            .success()
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn production_ramp_sets_linear_temperatures_and_records_endpoint() {
    use md::trajectory::{initial_state, read_run, rescale};
    use md::{Integrator, VelocityVerlet};
    let dir = std::env::temp_dir().join(format!("md-ramp-{}", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_md"))
        .args([
            "run",
            "--temperature",
            "0.5",
            "--ramp-to",
            "1.1",
            "--eq-steps",
            "51",
            "--steps",
            "6",
            "--sample-every",
            "1",
            "--out",
        ])
        .arg(&dir)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("run.json")).unwrap()).unwrap();
    assert_eq!(metadata["ramp_to"], 1.1);
    let (config, frames) = read_run(&dir).unwrap();
    assert_eq!(config.ramp_to, Some(1.1));
    for (frame, expected) in frames.iter().zip([0.6, 0.7, 0.8, 0.9, 1.0, 1.1]) {
        let kinetic = frame.state(&config).kinetic_energy();
        assert!((kinetic / (config.n - 1) as f64 - expected).abs() < 1e-12);
        assert!((frame.e_kin - kinetic).abs() < 1e-12);
    }
    // Equilibration ends between thermostat updates; production must start at T=0.5.
    let mut start = initial_state(&config).unwrap();
    for step in 1..=51 {
        VelocityVerlet.step(&mut start, config.dt).unwrap();
        if step == 50 {
            rescale(&mut start, 0.5).unwrap();
        }
    }
    rescale(&mut start, 0.5).unwrap();
    VelocityVerlet.step(&mut start, config.dt).unwrap();
    assert_eq!(frames[0].pos, start.positions);
    for target in [f64::NAN, f64::INFINITY, -1.0, 0.0, 0.4] {
        let mut invalid = config.clone();
        invalid.ramp_to = Some(target);
        assert!(invalid.validate().is_err());
    }
    fs::remove_dir_all(&dir).unwrap();
}
