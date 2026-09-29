"""Run with: uv run --no-project --with numpy python week5/scripts/test_seismic_forward.py"""

import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np


week = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temp:
    temp = Path(temp)
    experiment_path = temp / "experiment.json"
    output = temp / "output"
    experiment = {
        "schema": "week5-seismic-experiment-v1",
        "name": "small",
        "nx": 5,
        "nz": 5,
        "dx": 1.0,
        "dt": 0.1,
        "steps": 2,
        "source_frequency": 1.0,
        "source_peak_time": 0.0,
        "source_amplitude": 1.0,
        "sponge_width": 3,
        "sponge_strength": 0.2,
        "shots": [[2.0, 2.0], [1.0, 2.0]],
        "receivers": [[2, 2], [1, 2]],
        "background": [[1.0] * 5 for _ in range(5)],
        "perturbation": [[0.0] * 5 for _ in range(5)],
        "length_unit_m": 1.0,
        "time_unit_s": 1.0,
    }
    experiment_path.write_text(json.dumps(experiment))
    run = subprocess.run(
        ["cargo", "run", "--quiet", "--manifest-path", str(week / "seismic/Cargo.toml"), "--",
         "--experiment", str(experiment_path), "--mode", "forward", "--out", str(output),
         "--recording-every", "1"],
        check=True,
        capture_output=True,
        text=True,
    )

    traces = np.load(output / "traces.npy")
    frames = np.load(output / "wavefield.npy")
    assert traces.dtype == np.float64 and traces.shape == (2, 2, 2)
    assert frames.dtype == np.float32 and frames.shape == (2, 5, 5)

    dt = 0.1
    sigma_center = 0.2 * (1 - 2 / 3) ** 2
    sigma_left = 0.2 * (1 - 1 / 3) ** 2
    center_first = dt**2 / (1 + sigma_center * dt)
    left_first = dt**2 * math.exp(-0.5) / (1 + sigma_left * dt)
    pulse_second = (1 - 2 * (math.pi * dt) ** 2) * math.exp(-(math.pi * dt) ** 2)
    center_second = (2 * center_first + dt**2 * (4 * (left_first - center_first) + pulse_second)) / (1 + sigma_center * dt)
    np.testing.assert_allclose(traces[0, 0], [center_first, left_first], rtol=1e-14)
    np.testing.assert_allclose(traces[0, 1, 0], center_second, rtol=1e-14)
    np.testing.assert_allclose(traces[1, 0, 1], dt**2 / (1 + sigma_left * dt), rtol=1e-14)
    np.testing.assert_allclose(frames[:, 2, 2], traces[0, :, 0], rtol=1e-6)
    assert np.all(frames[:, 0, :] == 0) and np.all(frames[:, :, 0] == 0)

    metadata = json.loads((output / "run.json").read_text())
    assert metadata["experiment_file"] == str(experiment_path)
    assert metadata["experiment"]["name"] == "small"
    assert "background" not in metadata["experiment"]
    assert "perturbation" not in metadata["experiment"]
    assert metadata["recording"] == {"every": 1, "steps": [1, 2], "times": [0.1, 0.2]}
    result = json.loads((output / "result.json").read_text())
    assert result == {key: experiment[key] for key in ("nx", "nz", "dx", "dt", "steps", "shots", "receivers")} | {"mode": "forward"}
    lines = run.stdout.splitlines()
    assert lines[0] == "shot\tmode\tdata L2 norm" and len(lines) == 3
    for shot, line in enumerate(lines[1:]):
        index, mode, norm = line.split("\t")
        assert index == str(shot) and mode == "forward"
        assert math.isclose(float(norm), float(np.linalg.norm(traces[shot])), rel_tol=1e-12)
