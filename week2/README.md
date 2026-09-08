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
saved state at 30 fps. Generated files are excluded from Git.
