use crate::Result;
use crate::trajectory::{Frame, RunConfig};
pub fn energy_drift(energies: &[f64]) -> Result<f64> {
    if energies.is_empty() || energies[0] == 0.0 {
        return Err("undefined reference energy".into());
    }
    let k = (energies.len() / 10).max(1);
    let first = energies[..k].iter().sum::<f64>() / k as f64;
    let last = energies[energies.len() - k..].iter().sum::<f64>() / k as f64;
    let drift = (last - first).abs() / energies[0].abs();
    if !drift.is_finite() {
        return Err("nonfinite energy drift".into());
    }
    Ok(drift)
}
pub fn speed_statistics(speeds: &[f64]) -> Result<(f64, f64)> {
    let temperature = speeds.iter().map(|v| v * v).sum::<f64>() / (2.0 * speeds.len() as f64);
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err("undefined speed temperature".into());
    }
    let mut edges = [f64::INFINITY; 25];
    for (k, edge) in edges[..24].iter_mut().enumerate() {
        *edge = (-2.0 * temperature * (1.0 - k as f64 / 24.0).ln()).sqrt();
    }
    let mut counts = [0usize; 24];
    for speed in speeds {
        let bin = edges[1..24].partition_point(|edge| speed >= edge);
        counts[bin] += 1;
    }
    let expected = speeds.len() as f64 / 24.0;
    let shape = counts
        .iter()
        .map(|&n| (n as f64 - expected).powi(2) / expected)
        .sum::<f64>()
        / 22.0;
    Ok((temperature, shape))
}
pub struct Report {
    pub drift: f64,
    pub temperature: f64,
    pub speed_shape: f64,
}
impl Report {
    pub fn passed(&self, target: f64) -> bool {
        self.drift < 0.002 && (self.temperature - target).abs() < 0.05 && self.speed_shape < 2.0
    }
}
pub fn check(config: &RunConfig, frames: &[Frame]) -> Result<Report> {
    let energies = frames
        .iter()
        .map(|f| f.state(config).energy())
        .collect::<Result<Vec<_>>>()?;
    let speeds: Vec<f64> = frames
        .iter()
        .flat_map(|f| f.vel.iter().map(|v| v[0].hypot(v[1])))
        .collect();
    let (temperature, speed_shape) = speed_statistics(&speeds)?;
    Ok(Report {
        drift: energy_drift(&energies)?,
        temperature,
        speed_shape,
    })
}
pub struct Rdf {
    pub radius: Vec<f64>,
    pub values: Vec<f64>,
    pub contrast: f64,
}
pub fn rdf_series(config: &RunConfig, frames: &[Frame]) -> Vec<Rdf> {
    let rmax = config.box_size[0].min(config.box_size[1]) / 2.0;
    let width = rmax / 100.0;
    let radius: Vec<f64> = (0..100).map(|bin| (bin as f64 + 0.5) * width).collect();
    let mut counts = [0u64; 100];
    let mut curves = Vec::with_capacity(frames.len());
    for (index, frame) in frames.iter().enumerate() {
        let state = frame.state(config);
        for i in 0..config.n {
            for j in i + 1..config.n {
                let d = state.displacement(i, j);
                let r = d[0].hypot(d[1]);
                if r < rmax {
                    counts[(r / width).floor() as usize] += 2;
                }
            }
        }
        let values: Vec<f64> = (0..100)
            .map(|bin| {
                let inner = bin as f64 * width;
                let outer = (bin + 1) as f64 * width;
                counts[bin] as f64
                    / ((index + 1) as f64
                        * config.n as f64
                        * config.rho
                        * std::f64::consts::PI
                        * (outer * outer - inner * inner))
            })
            .collect();
        let long: Vec<f64> = radius
            .iter()
            .zip(&values)
            .filter(|(r, _)| **r > 2.0)
            .map(|(_, g)| (g - 1.0).powi(2))
            .collect();
        let contrast = (long.iter().sum::<f64>() / long.len() as f64).sqrt();
        curves.push(Rdf {
            radius: radius.clone(),
            values,
            contrast,
        });
    }
    curves
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{Frame, RunConfig, initial_state};
    #[test]
    fn drift_averages_end_windows_and_speed_shape_uses_22_degrees() {
        let mut energies = vec![-10.0; 20];
        energies[18] = -9.99;
        energies[19] = -9.99;
        assert!((energy_drift(&energies).unwrap() - 0.001).abs() < 1e-12);
        let (temperature, shape) = speed_statistics(&[1.0; 24]).unwrap();
        assert_eq!(temperature, 0.5);
        // All 24 speeds occupy one bin: (23² + 23)/22.
        assert!((shape - 552.0 / 22.0).abs() < 1e-12);
        assert!(energy_drift(&[0.0]).is_err());
        assert!(speed_statistics(&[0.0; 24]).is_err());
    }

    #[test]
    fn rdf_counts_both_neighbours_and_accumulates_frames() {
        // Synthetic pair isolates ring normalization; it is not a lattice initializer test.
        let config = RunConfig {
            n: 2,
            rho: 2.0 / 36.0,
            box_size: [6.0, 6.0],
            ..RunConfig::default()
        };
        let first = Frame {
            step: 50,
            t: 0.5,
            pos: vec![[0.0, 0.0], [1.0, 0.0]],
            vel: vec![[0.0, 0.0]; 2],
            e_pot: 0.0,
            e_kin: 0.0,
        };
        let mut second = first.clone();
        second.pos[1][0] = 1.5;
        let curves = rdf_series(&config, &[first, second]);
        let expected = 2.0
            / (2.0 * (2.0 / 36.0) * std::f64::consts::PI * (1.02_f64.powi(2) - 0.99_f64.powi(2)));
        assert!((curves[0].values[33] - expected).abs() < 1e-10);
        assert!((curves[1].values[33] - expected / 2.0).abs() < 1e-10);
        assert_eq!(curves[0].contrast, 1.0); // No pairs beyond radius 2: g=0 there.
    }
    #[test]
    fn physical_checks_ignore_stored_energy_fields() {
        let config = RunConfig::default();
        let state = initial_state(&config).unwrap();
        let frame = Frame {
            step: 50,
            t: 0.5,
            pos: state.positions,
            vel: state.velocities,
            e_pot: 0.0,
            e_kin: 0.0,
        };
        let mut frames = vec![frame.clone(), frame];
        let a = check(&config, &frames).unwrap();
        frames[0].e_pot = 99999.0;
        frames[1].e_kin = 123.0;
        let b = check(&config, &frames).unwrap();
        assert_eq!(a.drift, b.drift);
        assert_eq!(a.temperature, b.temperature);
        assert_eq!(a.speed_shape, b.speed_shape);
    }
}
