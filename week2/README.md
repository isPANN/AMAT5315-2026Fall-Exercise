# Week 2

Generate the dimer energy-error plot from the repository root:

```sh
cargo run --manifest-path week2/md/Cargo.toml --example dimer
```

## Lennard–Jones fluid

From `week2/`:

    make reproduce
    md/target/release/md check artifacts
    md/target/release/md video artifacts --out artifacts/run.mp4

Video requires ffmpeg with libx264. Install ffmpeg first (on macOS, `brew install ffmpeg`).
The default run uses 100 atoms, density 0.8, target temperature 0.5, dt 0.01,
2000 equilibration steps, 10000 production steps, sampling every 50 steps, and seed 2026.
It writes run.json and 200 production frames to traj.jsonl. Production step zero is not saved.

For options, run `md/target/release/md run --help`. To install the `md` command:

    cargo install --path md

The thermostat uses 2N-2 degrees of freedom during equilibration. Saved-speed temperature
uses mean(v²)/2. Check recomputes energy and speed statistics from saved states and exits
unsuccessfully if data are malformed or any of the three physical checks fails.
The video shows the cumulative 100-bin RDF, its long-range contrast, and one image per
saved state at 30 fps. Scratch outputs are excluded from Git; the published heating
trajectory, cold/hot videos, and scaling figure are committed.

## Cold and hot videos

From the repository root, reproduce [cold.mp4](cold.mp4) and [hot.mp4](hot.mp4).
Both use the default 100-atom run and save 200 frames; only the initial and
equilibration target temperature differs. Production has no thermostat.

```sh
md run --temperature 0.2 --out /tmp/cold
md video /tmp/cold --out week2/cold.mp4
md run --temperature 1.0 --out /tmp/hot
md video /tmp/hot --out week2/hot.mp4
```

## Heating run and Pages figures

From the repository root, reproduce the published heating trajectory (400 atoms,
200 saved frames, production target rising from 0.2 to 1.2):

```sh
md run --force cells --n 400 --rho 0.8 --temperature 0.2 --ramp-to 1.2 \
  --dt 0.01 --eq-steps 2000 --steps 20000 --sample-every 100 --seed 2026 --out docs
python3 -m http.server 8000 --directory docs
```

Open `http://localhost:8000/` to reproduce the particle, speed-distribution,
radial-distribution, and temperature/energy panels from this trajectory. Select a
frame and press **save PNG** to export those four panels. `md check` tests an
unheated equilibrium trajectory; its energy-conservation and fixed-temperature
gates do not apply to the heating run.

## Pages

[Open the molecular dynamics viewer](https://ispann.github.io/AMAT5315-2026Fall-Exercise/).

## Timing

From the repository root, build both Rust profiles, then print the median and
min–max of three complete executions per program (compilation is not timed):

```sh
cargo build --manifest-path week2/md/Cargo.toml --locked
cargo build --release --manifest-path week2/md/Cargo.toml --locked
uv run --with numpy python week2/timing.py
```

The Python command runs `week2-sim.py`; each Rust command runs `md run --out artifacts`
using its respective build executable. Every execution uses a fresh temporary working
directory. All use N=100, 2,000 equilibration steps, 10,000 production steps, and a
0.5 starting temperature. The Python reference uses seed 42 and all-pairs NumPy forces;
Rust uses its defaults (seed 2026, cell lists). This compares the supplied programs'
complete workloads, including startup and output; timings vary by machine and load.


  | Program | Median (s) | Range: min–max (s) |
  | --- | ---: | ---: |
  | NumPy week2-sim.py | 3.646 | 3.573–7.040 |
  | Rust debug | 7.360 | 7.323–7.975 |
  | Rust release | 0.383 | 0.381–0.894 |

## Benchmark

To rerun the Benchmark table and regenerate `scaling.png`, start from the repository
root (Rust, Python, and `uv` are required):

```sh
CARGO_PROFILE_RELEASE_DEBUG=true CARGO_PROFILE_RELEASE_STRIP=none \
CARGO_PROFILE_RELEASE_SPLIT_DEBUGINFO=packed RUSTFLAGS='-C force-frame-pointers=yes' \
  cargo install --path week2/md --locked --root "$HOME/.local" --force
export PATH="$HOME/.local/bin:$PATH"
cd week2
uv run --with matplotlib python benchmark.py
```

These build settings are for macOS. The script prints the table and draws both
labelled lines with range bars. Rerunning timing experiments produces new timings;
the numbers below are the recorded measurements, not exact-output requirements.

Three runs per force method and N on an Apple M3 (arm64, macOS 26.6.2), using the
installed optimized Rust build with debug symbols and frame pointers:

```sh
md run --force naive --n 100 --steps 500 --eq-steps 100 --out <unique-directory>
md run --force cells --n 100 --steps 500 --eq-steps 100 --out <unique-directory>
```

Repeat for N = 400 and 1600; all other parameters use their defaults (no temperature
ramp). Runs execute sequentially, alternating which method runs first in each pair.
Times are elapsed wall-clock seconds, including process startup and writing 10 frames.
Each entry is the median (min–max). Speedups are calculated as naive / cells within
each of the three trial pairs, then summarized by their median and range.

| N | naive (s) | cells (s) | speedup = naive / cells |
| ---: | ---: | ---: | ---: |
| 100 | 0.022118 (0.021863–0.026148) | 0.023242 (0.022624–0.023508) | 0.966× (0.941–1.125×) |
| 400 | 0.217425 (0.216590–0.217517) | 0.078558 (0.077767–0.080826) | 2.768× (2.691–2.785×) |
| 1600 | 3.089659 (3.080225–3.142572) | 0.344967 (0.341864–0.349475) | 8.956× (8.814–9.192×) |

The plot divides total wall time by all 600 integration steps (100 equilibration +
500 production). Both axes are logarithmic; error bars show min–max across three runs.

![Naive and cell-list timing versus particle count](scaling.png)
