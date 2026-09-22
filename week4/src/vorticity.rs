use rustfft::{Fft, FftPlanner, num_complex::Complex64};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize, Serialize)]
pub struct VelocityField {
    pub case: String,
    pub n: usize,
    pub seed: Option<u64>,
    pub k_band: Option<[i32; 2]>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
}

pub struct SpectralGrid {
    n: usize,
    cutoff: i32,
    forward_fft: Arc<dyn Fft<f64>>,
    inverse_fft: Arc<dyn Fft<f64>>,
}

impl SpectralGrid {
    pub fn new(n: usize) -> Self {
        assert!(n >= 3, "grid needs at least three points per side");
        assert!(n <= i32::MAX as usize, "grid dimension is too large");
        n.checked_mul(n).expect("grid is too large");
        let mut planner = FftPlanner::new();
        Self {
            n,
            cutoff: (n / 3) as i32,
            forward_fft: planner.plan_fft_forward(n),
            inverse_fft: planner.plan_fft_inverse(n),
        }
    }

    fn wave_number(&self, index: usize) -> i32 {
        if index <= (self.n - 1) / 2 {
            index as i32
        } else {
            index as i32 - self.n as i32
        }
    }

    fn transform_2d(&self, values: &mut [Complex64], fft: &Arc<dyn Fft<f64>>) {
        for row in values.chunks_exact_mut(self.n) {
            fft.process(row);
        }
        let mut column = vec![Complex64::default(); self.n];
        for x in 0..self.n {
            for y in 0..self.n {
                column[y] = values[y * self.n + x];
            }
            fft.process(&mut column);
            for y in 0..self.n {
                values[y * self.n + x] = column[y];
            }
        }
    }

    fn forward(&self, values: &[f64]) -> Vec<Complex64> {
        assert_eq!(values.len(), self.n * self.n, "field must be n by n");
        let mut spectrum: Vec<_> = values
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect();
        self.transform_2d(&mut spectrum, &self.forward_fft);
        spectrum
    }

    fn inverse(&self, mut spectrum: Vec<Complex64>) -> Vec<f64> {
        self.transform_2d(&mut spectrum, &self.inverse_fft);
        let scale = 1.0 / (self.n * self.n) as f64;
        spectrum.into_iter().map(|value| value.re * scale).collect()
    }

    fn truncate(&self, spectrum: &mut [Complex64]) {
        for y in 0..self.n {
            let ky = self.wave_number(y);
            for x in 0..self.n {
                let kx = self.wave_number(x);
                if kx.abs() > self.cutoff || ky.abs() > self.cutoff {
                    spectrum[y * self.n + x] = Complex64::default();
                }
            }
        }
    }

    fn filtered_spectrum(&self, values: &[f64]) -> Vec<Complex64> {
        let mut spectrum = self.forward(values);
        self.truncate(&mut spectrum);
        spectrum
    }

    pub fn filter(&self, values: &[f64]) -> Vec<f64> {
        self.inverse(self.filtered_spectrum(values))
    }

    pub fn derivatives(&self, values: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let spectrum = self.filtered_spectrum(values);
        let mut dx = spectrum.clone();
        let mut dxx = spectrum.clone();
        let mut dxdy = spectrum.clone();
        let mut laplacian = spectrum.clone();
        for y in 0..self.n {
            let ky = self.wave_number(y) as f64;
            for x in 0..self.n {
                let kx = self.wave_number(x) as f64;
                let i = y * self.n + x;
                let value = spectrum[i];
                dx[i] = Complex64::new(-kx * value.im, kx * value.re);
                dxx[i] = -kx * kx * value;
                dxdy[i] = -kx * ky * value;
                laplacian[i] = -(kx * kx + ky * ky) * value;
            }
        }
        (
            self.inverse(dx),
            self.inverse(dxx),
            self.inverse(dxdy),
            self.inverse(laplacian),
        )
    }

    pub fn velocity_from_vorticity(&self, omega: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let omega_hat = self.filtered_spectrum(omega);
        let mut u_hat = vec![Complex64::default(); omega_hat.len()];
        let mut v_hat = vec![Complex64::default(); omega_hat.len()];
        for y in 0..self.n {
            let ky = self.wave_number(y) as f64;
            for x in 0..self.n {
                let kx = self.wave_number(x) as f64;
                let i = y * self.n + x;
                let k_squared = kx * kx + ky * ky;
                if k_squared > 0.0 {
                    let psi = omega_hat[i] / k_squared;
                    u_hat[i] = Complex64::new(-ky * psi.im, ky * psi.re);
                    v_hat[i] = Complex64::new(kx * psi.im, -kx * psi.re);
                }
            }
        }
        (self.inverse(u_hat), self.inverse(v_hat))
    }

    pub fn vorticity_from_velocity(&self, u: &[f64], v: &[f64]) -> Vec<f64> {
        let u_hat = self.forward(u);
        let v_hat = self.forward(v);
        let mut omega_hat = vec![Complex64::default(); u_hat.len()];
        for y in 0..self.n {
            let ky = self.wave_number(y) as f64;
            for x in 0..self.n {
                let kx = self.wave_number(x) as f64;
                let i = y * self.n + x;
                let v_x = Complex64::new(-kx * v_hat[i].im, kx * v_hat[i].re);
                let u_y = Complex64::new(-ky * u_hat[i].im, ky * u_hat[i].re);
                omega_hat[i] = v_x - u_y;
            }
        }
        self.truncate(&mut omega_hat);
        self.inverse(omega_hat)
    }

    pub fn rate(&self, omega: &[f64], nu: f64) -> Vec<f64> {
        let omega_hat = self.filtered_spectrum(omega);
        let mut u_hat = vec![Complex64::default(); omega_hat.len()];
        let mut v_hat = vec![Complex64::default(); omega_hat.len()];
        let mut omega_x_hat = vec![Complex64::default(); omega_hat.len()];
        let mut omega_y_hat = vec![Complex64::default(); omega_hat.len()];

        for y in 0..self.n {
            let ky = self.wave_number(y) as f64;
            for x in 0..self.n {
                let kx = self.wave_number(x) as f64;
                let i = y * self.n + x;
                let value = omega_hat[i];
                omega_x_hat[i] = Complex64::new(-kx * value.im, kx * value.re);
                omega_y_hat[i] = Complex64::new(-ky * value.im, ky * value.re);
                let k_squared = kx * kx + ky * ky;
                if k_squared > 0.0 {
                    let psi = value / k_squared;
                    u_hat[i] = Complex64::new(-ky * psi.im, ky * psi.re);
                    v_hat[i] = Complex64::new(kx * psi.im, -kx * psi.re);
                }
            }
        }

        let u = self.inverse(u_hat);
        let v = self.inverse(v_hat);
        let omega_x = self.inverse(omega_x_hat);
        let omega_y = self.inverse(omega_y_hat);
        let product: Vec<_> = u
            .iter()
            .zip(v.iter())
            .zip(omega_x.iter().zip(omega_y.iter()))
            .map(|((&u, &v), (&omega_x, &omega_y))| u * omega_x + v * omega_y)
            .collect();
        let mut product_hat = self.forward(&product);
        self.truncate(&mut product_hat);

        for y in 0..self.n {
            let ky = self.wave_number(y) as f64;
            for x in 0..self.n {
                let kx = self.wave_number(x) as f64;
                let i = y * self.n + x;
                product_hat[i] = -product_hat[i] - nu * (kx * kx + ky * ky) * omega_hat[i];
            }
        }
        self.inverse(product_hat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn taylor_green_decays_at_its_exact_rate() {
        let n = 12;
        let grid = SpectralGrid::new(n);
        let omega: Vec<_> = (0..n)
            .flat_map(|y| {
                (0..n).map(move |x| {
                    -2.0 * (TAU * x as f64 / n as f64).cos() * (TAU * y as f64 / n as f64).cos()
                })
            })
            .collect();
        let rate = grid.rate(&omega, 0.1);
        let error = rate
            .iter()
            .zip(omega)
            .map(|(actual, omega)| (actual + 0.2 * omega).abs())
            .fold(0.0, f64::max);
        assert!(error < 1e-12, "maximum error: {error}");
    }

    #[test]
    fn filter_removes_modes_above_two_thirds_cutoff() {
        let n = 12;
        let grid = SpectralGrid::new(n);
        let values: Vec<_> = (0..n)
            .flat_map(|_| (0..n).map(|x| (5.0 * TAU * x as f64 / n as f64).cos()))
            .collect();
        assert!(grid.filter(&values).iter().all(|value| value.abs() < 1e-12));
    }
}
