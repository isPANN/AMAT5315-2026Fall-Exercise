use md::{ForwardEuler, VelocityVerlet, dimer_state, run};
use plotters::prelude::*;
use std::{error::Error, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let euler = run(&ForwardEuler, dimer_state(), 0.01, 500)?;
    let verlet = run(&VelocityVerlet, dimer_state(), 0.01, 5000)?;
    let initial_energy = dimer_state().energy()?.abs();
    let euler_points: Vec<_> = euler
        .iter()
        .map(|sample| (sample.time, sample.energy_error / initial_energy))
        .collect();
    let verlet_short: Vec<_> = verlet[..=500]
        .iter()
        .map(|sample| (sample.time, sample.energy_error / initial_energy))
        .collect();
    let verlet_long: Vec<_> = verlet
        .iter()
        .map(|sample| (sample.time, 1000.0 * sample.energy_error / initial_energy))
        .collect();

    let left_low = euler_points
        .iter()
        .chain(&verlet_short)
        .map(|point| point.1)
        .fold(0.0_f64, f64::min);
    let left_high = euler_points
        .iter()
        .chain(&verlet_short)
        .map(|point| point.1)
        .fold(0.0_f64, f64::max);
    let right_low = verlet_long
        .iter()
        .map(|point| point.1)
        .fold(0.0_f64, f64::min);
    let right_high = verlet_long
        .iter()
        .map(|point| point.1)
        .fold(0.0_f64, f64::max);

    let output = Path::new(env!("CARGO_MANIFEST_DIR")).join("../dimer.png");
    let root = BitMapBackend::new(&output, (1400, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, 2));

    let mut left = ChartBuilder::on(&panels[0])
        .caption("500 steps, Δt = 0.01", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(90)
        .build_cartesian_2d(
            0.0..5.0,
            left_low - 0.08 * (left_high - left_low)..left_high + 0.08 * (left_high - left_low),
        )?;
    left.configure_mesh()
        .max_light_lines(0)
        .x_desc("Time")
        .y_desc("(E(t) - E0) / |E0|")
        .draw()?;
    for (label, points, color) in [
        ("Forward Euler", euler_points, RED),
        ("Velocity-Verlet", verlet_short, BLUE),
    ] {
        left.draw_series(std::iter::once(PathElement::new(
            points,
            color.stroke_width(2),
        )))?
        .label(label)
        .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], color));
    }
    left.configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .draw()?;

    let mut right = ChartBuilder::on(&panels[1])
        .caption("Velocity-Verlet, 5000 steps", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(90)
        .build_cartesian_2d(
            0.0..50.0,
            right_low - 0.08 * (right_high - right_low)
                ..right_high + 0.08 * (right_high - right_low),
        )?;
    right
        .configure_mesh()
        .max_light_lines(0)
        .x_desc("Time")
        .y_desc("1000 × (E(t) - E0) / |E0|")
        .draw()?;
    right.draw_series(std::iter::once(PathElement::new(
        verlet_long,
        BLUE.stroke_width(2),
    )))?;

    root.present()?;
    println!("{}", output.display());
    Ok(())
}
