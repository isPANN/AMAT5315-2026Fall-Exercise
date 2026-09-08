use crate::{
    Result,
    analysis::Rdf,
    trajectory::{Frame, RunConfig},
};
use plotters::prelude::*;
use std::path::Path;
fn render_frame(
    config: &RunConfig,
    frame: &Frame,
    rdf: &Rdf,
    ymax: f64,
    path: &Path,
) -> Result<()> {
    let root = BitMapBackend::new(path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let (atoms, structure) = root.split_horizontally(600);
    atoms.draw(&Text::new(
        format!("Lennard-Jones fluid | t = {:.2}", frame.t),
        (30, 30),
        ("sans-serif", 24),
    ))?;
    let scale = 500.0 / config.box_size[0];
    let top = (550.0 - config.box_size[1] * scale).round() as i32;
    atoms.draw(&Rectangle::new([(50, top), (550, 550)], BLACK))?;
    for pos in &frame.pos {
        atoms.draw(&Circle::new(
            (
                (50.0 + pos[0] * scale).round() as i32,
                (550.0 - pos[1] * scale).round() as i32,
            ),
            3,
            BLUE.filled(),
        ))?;
    }
    atoms.draw(&Text::new(
        format!(
            "N = {} | box {:.3} × {:.3}",
            config.n, config.box_size[0], config.box_size[1]
        ),
        (50, 580),
        ("sans-serif", 18),
    ))?;
    let mut chart = ChartBuilder::on(&structure)
        .caption("Cumulative radial distribution", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(65)
        .build_cartesian_2d(
            0.0..config.box_size[0].min(config.box_size[1]) / 2.0,
            0.0..ymax,
        )?;
    chart
        .configure_mesh()
        .max_light_lines(0)
        .x_desc("r / sigma")
        .y_desc("g(r)")
        .axis_desc_style(("sans-serif", 18))
        .label_style(("sans-serif", 16))
        .draw()?;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 1.0), (config.box_size[1] / 2.0, 1.0)],
        BLACK.mix(0.4),
    )))?;
    let points: Vec<_> = rdf
        .radius
        .iter()
        .copied()
        .zip(rdf.values.iter().copied())
        .collect();
    chart.draw_series(std::iter::once(PathElement::new(
        points,
        RED.stroke_width(2),
    )))?;
    atoms.draw(&Text::new(
        format!("Long-range contrast: {:.4}", rdf.contrast),
        (30, 70),
        ("sans-serif", 18),
    ))?;
    root.present()?;
    Ok(())
}
pub fn record(config: &RunConfig, frames: &[Frame], out: &Path) -> Result<()> {
    let curves = crate::analysis::rdf_series(config, frames);
    let ymax = 1.08
        * curves
            .iter()
            .flat_map(|r| r.values.iter())
            .copied()
            .fold(1.0_f64, f64::max);
    let temp = std::env::temp_dir().join(format!("md-encode-{}", std::process::id()));
    std::fs::create_dir(&temp)?;
    for (index, (frame, rdf)) in frames.iter().zip(&curves).enumerate() {
        render_frame(
            config,
            frame,
            rdf,
            ymax,
            &temp.join(format!("{index:06}.png")),
        )?;
    }
    let bitrate = (1_600_000_u64 * 8 * 30 / frames.len() as u64).to_string();
    let input = temp.join("%06d.png");
    let log = temp.join("x264");
    for pass in [1, 2] {
        let mut command = std::process::Command::new("ffmpeg");
        command
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                "30",
                "-start_number",
                "0",
                "-i",
            ])
            .arg(&input)
            .args(["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v"])
            .arg(&bitrate)
            .args(["-pass", &pass.to_string(), "-passlogfile"])
            .arg(&log);
        if pass == 1 {
            command.args(["-f", "null", "/dev/null"]);
        } else {
            command.args(["-movflags", "+faststart"]).arg(out);
        }
        let status = command.status()?;
        if !status.success() {
            return Err(format!("ffmpeg pass {pass} failed; frames at {}", temp.display()).into());
        }
    }
    if std::fs::metadata(out)?.len() >= 2_000_000 {
        return Err(format!("MP4 exceeds size limit; frames at {}", temp.display()).into());
    }
    std::fs::remove_dir_all(&temp)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::RunConfig;
    #[test]
    #[ignore = "requires ffmpeg and ffprobe"]
    fn video_preserves_saved_frame_count_and_size_limit() {
        let dir = std::env::temp_dir().join(format!("md-video-check-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let config = RunConfig {
            eq_steps: 0,
            steps: 20,
            sample_every: 10,
            ..RunConfig::default()
        };
        crate::trajectory::write_run(&config, &dir).unwrap();
        let (config, frames) = crate::trajectory::read_run(&dir).unwrap();
        let output = dir.join("run.mp4");
        record(&config, &frames, &output).unwrap();
        assert!(std::fs::metadata(&output).unwrap().len() < 2_000_000);
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
            ])
            .arg(&output)
            .output()
            .unwrap();
        assert!(probe.status.success());
        assert_eq!(String::from_utf8(probe.stdout).unwrap().trim(), "2");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
