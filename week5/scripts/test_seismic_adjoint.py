"""Run with: uv run --no-project --with numpy python week5/scripts/test_seismic_adjoint.py"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


week = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    first = np.zeros((7, 7))
    first[2:5, 2:5] = [[0.1, -0.2, 0.3], [0.4, 0.7, -0.1], [-0.2, 0.3, 0.2]]
    second = np.zeros((7, 7))
    second[2:5, 2:5] = [[-0.3, 0.2, 0.1], [0.1, 0.5, 0.4], [0.2, -0.1, 0.3]]
    experiment = {
        "nx": 7, "nz": 7, "dx": 1.0, "dt": 0.15, "steps": 7,
        "source_frequency": 1.0, "source_peak_time": 0.0,
        "source_amplitude": 1.0, "sponge_width": 3, "sponge_strength": 0.2,
        "shots": [[3.0, 3.0], [2.0, 3.0]],
        "receivers": [[3, 3], [2, 3], [4, 4]],
        "background": [[1.2] * 7 for _ in range(7)],
    }

    def run(mode, perturbation, name, data=None, every=2, speed=None,
            storage=None, checkpoints=None):
        path = directory / f"{name}.json"
        path.write_text(json.dumps(experiment | {
            "perturbation": perturbation.tolist(),
            "background": (speed if speed is not None else np.full((7, 7), 1.2)).tolist(),
        }))
        output = directory / name
        command = [
            "cargo", "run", "--quiet", "--manifest-path", str(week / "seismic/Cargo.toml"), "--",
            "--experiment", str(path), "--mode", mode, "--out", str(output),
        ]
        if data is not None:
            command += ["--data", str(data), "--every", str(every)]
        if storage is not None:
            command += ["--storage", storage]
        if checkpoints is not None:
            command += ["--checkpoints", str(checkpoints)]
        completed = subprocess.run(command, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        return output, completed

    first_output, _ = run("born", first, "first")
    second_output, _ = run("born", second, "second")
    born = np.load(first_output / "born_data.npy")
    weights = np.load(second_output / "born_data.npy")
    output, completed = run("adjoint", first, "adjoint", second_output / "born_data.npy")
    image = np.load(output / "image.npy")
    assert image.dtype == np.float64 and image.shape == (7, 7)
    assert np.max(np.abs(image)) > 1e-5
    np.testing.assert_allclose(np.sum(born * weights), np.sum(first * image), rtol=1e-10, atol=1e-11)
    epsilon = 1e-5
    speed_plus = np.full((7, 7), 1.2)
    speed_minus = speed_plus.copy()
    speed_plus[3, 3] += epsilon
    speed_minus[3, 3] -= epsilon
    plus_output, _ = run("forward", first, "plus", speed=speed_plus)
    minus_output, _ = run("forward", first, "minus", speed=speed_minus)
    plus = np.load(plus_output / "traces.npy")
    minus = np.load(minus_output / "traces.npy")
    finite_difference = np.sum(weights * (plus - minus)) / (2 * epsilon)
    np.testing.assert_allclose(image[3, 3], finite_difference, rtol=2e-6, atol=1e-9)

    frames = np.load(output / "wavefield.npy")
    assert frames.dtype == np.float32 and frames.shape == (3, 7, 7)
    assert np.isfinite(frames).all() and np.max(np.abs(frames)) > 0
    recording = json.loads((output / "run.json").read_text())["recording"]
    assert recording["every"] == 2 and recording["steps"] == [6, 4, 2]
    np.testing.assert_allclose(recording["times"], [0.9, 0.6, 0.3], rtol=0, atol=1e-15)
    terminal_output, _ = run(
        "adjoint", first, "terminal", second_output / "born_data.npy", every=1,
    )
    terminal_frame = np.load(terminal_output / "wavefield.npy")[0]
    expected_terminal = np.zeros((7, 7))
    for receiver, (x, z) in enumerate(experiment["receivers"]):
        expected_terminal[z, x] += weights[0, -1, receiver]
    np.testing.assert_allclose(terminal_frame, expected_terminal, rtol=1e-6, atol=1e-10)
    result = json.loads((output / "result.json").read_text())
    assert result["mode"] == "adjoint"
    assert result["statistics"] == {
        "storage": "full", "checkpoints": None,
        "reverse_calls": 14, "scheduler_forward_calls": 14,
        "peak_saved_states": 8, "peak_saved_bytes": 8 * 2 * 7 * 7 * 8,
        "per_shot": [
            {"reverse_calls": 7, "scheduler_forward_calls": 7, "peak_saved_states": 8},
            {"reverse_calls": 7, "scheduler_forward_calls": 7, "peak_saved_states": 8},
        ],
    }
    lines = completed.stdout.splitlines()
    assert lines[0] == "shot\tmode\tdata L2 norm" and len(lines) == 3
    for shot, line in enumerate(lines[1:]):
        index, mode, norm = line.split("\t")
        assert index == str(shot) and mode == "adjoint"
        assert np.isclose(float(norm), np.linalg.norm(weights[shot]), rtol=1e-12)

    for budget in (1, 2):
        tree_output, _ = run(
            "adjoint", first, f"treeverse_{budget}", second_output / "born_data.npy",
            storage="treeverse", checkpoints=budget,
        )
        np.testing.assert_array_equal(np.load(tree_output / "image.npy"), image)
        np.testing.assert_array_equal(np.load(tree_output / "wavefield.npy"), frames)
        statistics = json.loads((tree_output / "result.json").read_text())["statistics"]
        assert statistics["storage"] == "treeverse"
        assert statistics["checkpoints"] == budget
        assert statistics["reverse_calls"] == 14
        assert statistics["peak_saved_states"] <= budget + 1
        assert statistics["peak_saved_bytes"] == statistics["peak_saved_states"] * 2 * 7 * 7 * 8
        forward_calls = 0
        for shot, shot_stats in enumerate(statistics["per_shot"]):
            assert shot_stats["actions_file"] == f"actions-{shot}.json"
            actions = json.loads((tree_output / shot_stats["actions_file"]).read_text())
            saved = {0}
            working = 0
            reversed_steps = []
            shot_calls = 0
            peak_saved = 1
            for entry in actions:
                action, step = entry["action"], entry["step"]
                if action == "restore":
                    assert step in saved
                    working = step
                elif action == "call":
                    assert step == working
                    working += 1
                    shot_calls += 1
                elif action == "store":
                    assert step == working and step not in saved
                    saved.add(step)
                elif action == "grad":
                    assert step in saved
                    reversed_steps.append(step)
                elif action == "fetch":
                    assert step in saved and step != 0
                    saved.remove(step)
                else:
                    raise AssertionError(f"unknown action {action}")
                assert entry["saved_states"] == len(saved) <= budget + 1
                peak_saved = max(peak_saved, len(saved))
            assert reversed_steps == list(range(6, -1, -1))
            assert saved == {0}
            assert shot_stats["reverse_calls"] == 7
            assert shot_stats["scheduler_forward_calls"] == shot_calls
            assert shot_stats["peak_saved_states"] == peak_saved
            forward_calls += shot_calls
        assert statistics["scheduler_forward_calls"] == forward_calls
        assert statistics["peak_saved_states"] == max(
            shot["peak_saved_states"] for shot in statistics["per_shot"]
        )
        if budget == 1:
            assert forward_calls == 42
        else:
            assert forward_calls == 24 and statistics["peak_saved_states"] == 3
