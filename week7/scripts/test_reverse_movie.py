import csv
import json
import subprocess
import sys

from scripts.noise_model import ddpm_trajectory, initialize_model
from scripts.reverse_movie import mode_statistics, reverse_frames

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def test_mode_statistics_use_nearest_centres_counts_and_population_spread():
    samples = np.array([[2., -0.5], [4., 0.5], [0., 3.],
                        [-4., -1.], [-3., 0.], [-2., 1.], [0., -3.], [0.5, -3.5]])
    labels, rows = mode_statistics(samples)
    np.testing.assert_array_equal(labels, [0, 0, 1, 2, 2, 2, 3, 3])
    np.testing.assert_array_equal([r["count"] for r in rows], [2, 1, 3, 2])
    np.testing.assert_allclose([r["fraction"] for r in rows], [0.25, 0.125, 0.375, 0.25])
    np.testing.assert_allclose([[r["mean_x1"], r["mean_x2"]] for r in rows], [[3., 0.], [0., 3.], [-3., 0.], [0.25, -3.25]])
    np.testing.assert_allclose([[r["variance_x1"], r["variance_x2"]] for r in rows], [[1., 0.25], [0., 0.], [2/3, 2/3], [0.0625, 0.0625]])
    np.testing.assert_allclose([[r["std_x1"], r["std_x2"]] for r in rows], [[1., 0.5], [0., 0.], [np.sqrt(2/3)] * 2, [0.25, 0.25]])


def test_reverse_frames_follow_saved_samples_with_fixed_axes_and_no_annotations():
    path = np.zeros((1001, 4, 2))
    path[0] = [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]
    path[500] = [[2., 0.], [0., 2.], [-2., 0.], [0., -2.]]
    path[-1] = [[3., 0.], [0., 3.], [-3., 0.], [0., -3.]]
    fig, update = reverse_frames(path)
    axes = fig.axes[0]
    limits = (axes.get_xlim(), axes.get_ylim())
    points, title = update(0)
    colours = points.get_facecolors().copy()
    for frame in [0, 500, 1000]:
        update(frame)
        np.testing.assert_array_equal(points.get_offsets(), path[frame])
        np.testing.assert_array_equal(points.get_facecolors(), colours)
        assert f"t = {1000 - frame}" in title.get_text()
        assert (axes.get_xlim(), axes.get_ylim()) == limits
        assert not axes.texts
    plt.close(fig)


def test_reverse_command_saves_full_ddpm_run_metrics_and_decodable_movie(tmp_path):
    params = jax.tree.map(jnp.zeros_like, initialize_model(jax.random.key(41)))
    checkpoint = tmp_path / "checkpoint.npz"
    np.savez(checkpoint, **params)
    run, figure, movie = tmp_path / "artifacts", tmp_path / "evidence" / "samples.png", tmp_path / "evidence" / "reverse.mp4"
    subprocess.run([sys.executable, "-m", "scripts.reverse_movie", "--checkpoint", str(checkpoint),
                    "--samples", "128", "--seed", "42", "--frame-stride", "1000",
                    "--output", str(run), "--figure", str(figure), "--movie", str(movie)],
                   check=True, capture_output=True, text=True)
    settings = json.loads((run / "settings.json").read_text())
    assert settings["samples"] == 128 and settings["ddpm_steps"] == 1000
    assert settings["backend"] == "cpu"
    assert settings["frame_levels"] == [1000, 0]
    assert settings["variance_ddof"] == 0
    key = jax.random.wrap_key_data(jnp.array(settings["keys"]["sampling"], dtype=jnp.uint32))
    with np.load(run / "trajectory.npz") as saved:
        trajectory = saved["trajectory"]
        assert trajectory.shape == (1001, 128, 2) and np.isfinite(trajectory).all()
        np.testing.assert_array_equal(saved["levels"], np.arange(1000, -1, -1))
        np.testing.assert_array_equal(trajectory, ddpm_trajectory(params, key, 128))
    with (run / "mode-statistics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert sum(int(row["count"]) for row in rows) == 128
    centres = np.array([[3., 0.], [0., 3.], [-3., 0.], [0., -3.]])
    labels = ((trajectory[-1, :, None] - centres) ** 2).sum(-1).argmin(1)
    for k, row in enumerate(rows):
        group = trajectory[-1][labels == k].astype(np.float64)
        np.testing.assert_allclose([float(row["mean_x1"]), float(row["mean_x2"])], group.mean(0))
        np.testing.assert_allclose([float(row["std_x1"]), float(row["std_x2"])], group.std(0))
    from PIL import Image
    with Image.open(figure) as image:
        image.verify()
    info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
                           "-of", "json", str(movie)], check=True, capture_output=True, text=True)
    assert int(json.loads(info.stdout)["streams"][0]["nb_frames"]) == 2
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(movie), "-f", "null", "-"], check=True, capture_output=True)
