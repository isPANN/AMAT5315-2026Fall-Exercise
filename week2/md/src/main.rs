use md::{ForwardEuler, VelocityVerlet, dimer_state, run};
use plotters::prelude::*;
use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

fn main() -> Result<(), Box<dyn Error>> {
    let euler = run(&ForwardEuler, dimer_state(), 0.01, 500)?;
    let verlet = run(&VelocityVerlet, dimer_state(), 0.01, 5000)?;
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let csv_path = directory.join("energy_error.csv");
    let png_path = directory.join("energy_error.png");
    let mut csv = BufWriter::new(File::create(&csv_path)?);
    writeln!(csv, "method,step,time,total_energy,energy_error")?;
    for (method, samples) in [("euler", &euler), ("verlet", &verlet)] {
        for sample in samples {
            writeln!(
                csv,
                "{},{},{},{},{}",
                method, sample.step, sample.time, sample.total_energy, sample.energy_error
            )?;
        }
    }
    csv.flush()?;

    let root = BitMapBackend::new(&png_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, 1));
    for (panel_index, panel) in panels.iter().enumerate() {
        let series = if panel_index == 0 {
            vec![
                ("Forward Euler", euler.as_slice(), RED),
                ("Velocity-Verlet", &verlet[..=500], BLUE),
            ]
        } else {
            vec![("Velocity-Verlet", verlet.as_slice(), BLUE)]
        };
        let mut low = 0.0_f64;
        let mut high = 0.0_f64;
        for (_, samples, _) in &series {
            for sample in *samples {
                low = low.min(sample.energy_error);
                high = high.max(sample.energy_error);
            }
        }
        let padding = 0.08 * (high - low);
        let (title, end_time) = if panel_index == 0 {
            ("Euler and velocity-Verlet: first 500 steps", 5.0)
        } else {
            ("Velocity-Verlet: 5000 steps", 50.0)
        };
        let mut chart = ChartBuilder::on(panel)
            .caption(title, ("sans-serif", 24))
            .margin(20)
            .x_label_area_size(50)
            .y_label_area_size(110)
            .build_cartesian_2d(0.0..end_time, (low - padding)..(high + padding))?;
        chart
            .configure_mesh()
            .max_light_lines(0)
            .x_labels(11)
            .y_labels(7)
            .axis_desc_style(("sans-serif", 18))
            .label_style(("sans-serif", 16))
            .x_desc("Time (reduced units)")
            .y_desc("E(t) - E0 (reduced energy)")
            .y_label_formatter(&|value| format!("{value:.5}"))
            .draw()?;
        for (label, samples, color) in series {
            let points: Vec<_> = samples
                .iter()
                .map(|sample| (sample.time, sample.energy_error))
                .collect();
            chart
                .draw_series(std::iter::once(PathElement::new(
                    points,
                    color.stroke_width(2),
                )))?
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }
        if panel_index == 0 {
            chart
                .configure_series_labels()
                .position(SeriesLabelPosition::UpperLeft)
                .label_font(("sans-serif", 18))
                .background_style(WHITE.mix(0.9))
                .border_style(BLACK)
                .draw()?;
        }
    }
    root.present()?;
    println!("{}\n{}", csv_path.display(), png_path.display());
    Ok(())
}
