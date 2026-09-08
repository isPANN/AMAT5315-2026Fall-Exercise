use plotters::prelude::*;
use std::{error::Error, path::Path};

fn energy_color(energy: f64) -> RGBColor {
    let strength = energy.abs().min(1.0);
    let pale = (255.0 * (1.0 - strength)) as u8;
    if energy < 0.0 {
        RGBColor(pale, pale, 255)
    } else {
        RGBColor(255, pale, pale)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let output = Path::new(env!("CARGO_MANIFEST_DIR")).join("../field.png");
    let root = BitMapBackend::new(&output, (900, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let (field, legend) = root.split_horizontally(800);
    let mut chart = ChartBuilder::on(&field)
        .caption("Lennard-Jones pair energy and force", ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-3.0..3.0, -3.0..3.0)?;
    chart
        .configure_mesh()
        .x_desc("x / σ")
        .y_desc("y / σ")
        .draw()?;

    let cells = 150;
    let step = 6.0 / cells as f64;
    chart.draw_series((0..cells).flat_map(|i| {
        (0..cells).map(move |j| {
            let x = -3.0 + (i as f64 + 0.5) * step;
            let y = -3.0 + (j as f64 + 0.5) * step;
            let r = x.hypot(y);
            Rectangle::new(
                [
                    (x - step / 2.0, y - step / 2.0),
                    (x + step / 2.0, y + step / 2.0),
                ],
                energy_color(md::lennard_jones_energy(r)).filled(),
            )
        })
    }))?;

    for i in -8..=8 {
        for j in -8..=8 {
            let (x, y) = (i as f64 * 0.35, j as f64 * 0.35);
            let r = x.hypot(y);
            if r < 0.7 {
                continue;
            }
            let force = md::lennard_jones_force(r);
            let length = 0.22 * (force.abs() / 2.0).min(1.0);
            let (dx, dy) = (
                length * force.signum() * x / r,
                length * force.signum() * y / r,
            );
            let end = (x + dx, y + dy);
            let angle = dy.atan2(dx);
            let head = (0.45 * length).min(0.06);
            chart.draw_series(std::iter::once(PathElement::new(
                vec![
                    (x, y),
                    end,
                    (
                        end.0 - head * (angle - 0.6).cos(),
                        end.1 - head * (angle - 0.6).sin(),
                    ),
                    end,
                    (
                        end.0 - head * (angle + 0.6).cos(),
                        end.1 - head * (angle + 0.6).sin(),
                    ),
                ],
                BLACK,
            )))?;
        }
    }
    chart.draw_series(std::iter::once(Circle::new((0.0, 0.0), 7, BLACK.filled())))?;

    legend.draw(&Text::new("U(r)", (20, 65), ("sans-serif", 18)))?;
    for y in 80..=680 {
        let energy = 1.0 - 2.0 * (y - 80) as f64 / 600.0;
        legend.draw(&Rectangle::new(
            [(20, y), (45, y + 1)],
            energy_color(energy).filled(),
        ))?;
    }
    for (label, y) in [("+1", 85), ("0", 385), ("−1", 685)] {
        legend.draw(&Text::new(label, (52, y), ("sans-serif", 15)))?;
    }
    legend.draw(&Text::new("ε", (52, 720), ("sans-serif", 15)))?;
    root.present()?;
    println!("{}", output.display());
    Ok(())
}
