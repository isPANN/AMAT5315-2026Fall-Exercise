import json
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.forward_movie import forward_frames


def test_frames_follow_saved_points_with_fixed_limits_and_component_colours():
    path = np.array([[[3., 0.], [0., 3.], [-3., 0.], [0., -3.]],
                     [[1., 1.], [-2., -1.], [0.5, 0.5], [-1., 2.]],
                     [[-1., 2.], [3., -2.], [1., 0.], [5., 4.]]])
    fig, update = forward_frames(path, np.array([0, 500, 1000]))
    points, title = update(0)
    colours = points.get_facecolors().copy()
    assert len(np.unique(colours, axis=0)) == 4
    limits = (fig.axes[0].get_xlim(), fig.axes[0].get_ylim())
    for frame in [0, 1, 2]:
        update(frame)
        np.testing.assert_array_equal(points.get_offsets(), path[frame])
        np.testing.assert_array_equal(points.get_facecolors(), colours)
        assert (fig.axes[0].get_xlim(), fig.axes[0].get_ylim()) == limits
        assert str([0, 500, 1000][frame]) in title.get_text()
    assert limits[0] == limits[1]
    assert limits[0][0] < path.min() and limits[0][1] > path.max()
    plt.close(fig)


def test_movie_command_encodes_saved_levels_and_prints_terminal_population_moments(tmp_path):
    run = tmp_path / "artifacts"
    run.mkdir()
    trajectory = np.array([[[3., 0.], [0., 3.], [-3., 0.], [0., -3.]],
                           [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]],
                           [[-1., 2.], [3., -2.], [1., 0.], [5., 4.]]])
    np.savez_compressed(run / "forward.npz", trajectory=trajectory, levels=[0, 500, 1000])
    output = tmp_path / "evidence" / "forward.mp4"
    result = subprocess.run([sys.executable, "-m", "scripts.forward_movie", "--run", str(run),
                             "--output", str(output)], check=True, capture_output=True, text=True)
    assert "t=1000 x1 mean=2.000000 variance=5.000000" in result.stdout
    assert "t=1000 x2 mean=1.000000 variance=5.000000" in result.stdout
    metadata = json.loads((run / "movie.json").read_text())
    assert metadata["levels"] == [0, 500, 1000]
    np.testing.assert_array_equal(metadata["terminal_mean"], [2., 1.])
    np.testing.assert_array_equal(metadata["terminal_variance"], [5., 5.])
    assert metadata["variance_ddof"] == 0
    info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                           "-show_entries", "stream=width,height,nb_frames,avg_frame_rate", "-of", "json", str(output)],
                          check=True, capture_output=True, text=True)
    stream = json.loads(info.stdout)["streams"][0]
    assert int(stream["nb_frames"]) == 3
    assert stream["avg_frame_rate"] == "30/1"
    assert stream["width"] == stream["height"] == 800
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"], check=True, capture_output=True)
