"""Run with: uv run --no-project --with numpy python week5/scripts/test_seismic_born.py"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


week = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    background = np.full((7, 7), 1.2)
    perturbation = np.zeros((7, 7))
    perturbation[2:5, 2:5] = [[0.1, -0.2, 0.3], [0.4, 0.7, -0.1], [-0.2, 0.3, 0.2]]
    experiment = {
        "nx": 7, "nz": 7, "dx": 1.0, "dt": 0.15, "steps": 8,
        "source_frequency": 1.0, "source_peak_time": 0.0,
        "source_amplitude": 1.0, "sponge_width": 3, "sponge_strength": 0.2,
        "shots": [[3.0, 3.0], [2.0, 3.0]],
        "receivers": [[3, 3], [2, 3], [4, 4]],
        "background": background.tolist(), "perturbation": perturbation.tolist(),
    }

    def run(mode, speed, name):
        path = directory / f"{name}.json"
        path.write_text(json.dumps(experiment | {"background": speed.tolist()}))
        output = directory / name
        command = [
            "cargo", "run", "--quiet", "--manifest-path", str(week / "seismic/Cargo.toml"), "--",
            "--experiment", str(path), "--mode", mode, "--out", str(output),
        ]
        completed = subprocess.run(
            command,
            capture_output=True, text=True,
        )
        assert completed.returncode == 0, completed.stderr
        data = np.load(output / ("born_data.npy" if mode == "born" else "traces.npy"))
        return data, completed, output

    born, completed, output = run("born", background, "born")
    epsilon = 1e-4
    plus, _, _ = run("forward", background + epsilon * perturbation, "plus")
    minus, _, _ = run("forward", background - epsilon * perturbation, "minus")
    reference = (plus - minus) / (2 * epsilon)
    assert born.dtype == np.float64 and born.shape == (2, 8, 3)
    assert np.max(np.abs(reference)) > 1e-4
    np.testing.assert_allclose(born, reference, rtol=2e-5, atol=2e-9)
    np.testing.assert_array_equal(born[:, 0], 0)
    assert not (output / "wavefield.npy").exists()
    rejected = subprocess.run(
        ["cargo", "run", "--quiet", "--manifest-path", str(week / "seismic/Cargo.toml"), "--",
         "--experiment", str(directory / "born.json"), "--mode", "born",
         "--out", str(directory / "rejected"), "--every", "2"],
        capture_output=True, text=True,
    )
    assert rejected.returncode != 0
    assert json.loads((output / "result.json").read_text())["mode"] == "born"
    lines = completed.stdout.splitlines()
    assert lines[0] == "shot\tmode\tdata L2 norm" and len(lines) == 3
    for shot, line in enumerate(lines[1:]):
        index, mode, norm = line.split("\t")
        assert index == str(shot) and mode == "born"
        assert np.isclose(float(norm), np.linalg.norm(born[shot]), rtol=1e-12)
