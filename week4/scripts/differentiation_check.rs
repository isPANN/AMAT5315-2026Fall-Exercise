use std::f64::consts::TAU;
use time_stepping::vorticity::SpectralGrid;

const NAMES: [&str; 4] = ["dx", "dxx", "dxdy", "laplacian"];

fn fields(n: usize) -> (Vec<f64>, [Vec<f64>; 4]) {
    let mut values = Vec::with_capacity(n * n);
    let mut exact: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(n * n));
    for y in 0..n {
        let y = TAU * y as f64 / n as f64;
        for x in 0..n {
            let x = TAU * x as f64 / n as f64;
            let sin_x = (3.0 * x).sin();
            let cos_x = (3.0 * x).cos();
            let sin_y = (2.0 * y).sin();
            let cos_y = (2.0 * y).cos();
            values.push(sin_x * cos_y);
            exact[0].push(3.0 * cos_x * cos_y);
            exact[1].push(-9.0 * sin_x * cos_y);
            exact[2].push(-6.0 * cos_x * sin_y);
            exact[3].push(-13.0 * sin_x * cos_y);
        }
    }
    (values, exact)
}

fn errors(actual: [Vec<f64>; 4], exact: &[Vec<f64>; 4]) -> [f64; 4] {
    std::array::from_fn(|derivative| {
        actual[derivative]
            .iter()
            .zip(exact[derivative].iter())
            .map(|(actual, exact)| (actual - exact).abs())
            .fold(0.0, f64::max)
    })
}

fn centered(n: usize) -> [f64; 4] {
    let (values, exact) = fields(n);
    let spacing = TAU / n as f64;
    let mut actual: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; n * n]);
    for y in 0..n {
        let down = (y + n - 1) % n;
        let up = (y + 1) % n;
        for x in 0..n {
            let left = (x + n - 1) % n;
            let right = (x + 1) % n;
            let i = y * n + x;
            actual[0][i] = (values[y * n + right] - values[y * n + left]) / (2.0 * spacing);
            actual[1][i] =
                (values[y * n + right] - 2.0 * values[i] + values[y * n + left]) / spacing.powi(2);
            actual[2][i] =
                (values[up * n + right] - values[down * n + right] - values[up * n + left]
                    + values[down * n + left])
                    / (4.0 * spacing.powi(2));
            actual[3][i] = actual[1][i]
                + (values[up * n + x] - 2.0 * values[i] + values[down * n + x]) / spacing.powi(2);
        }
    }
    errors(actual, &exact)
}

fn main() {
    let (values, exact) = fields(32);
    let (dx, dxx, dxdy, laplacian) = SpectralGrid::new(32).derivatives(&values);
    let fourier = errors([dx, dxx, dxdy, laplacian], &exact);
    let centered_32 = centered(32);
    let centered_64 = centered(64);

    println!("method\tn\tderivative\tmaximum absolute error");
    for (name, error) in NAMES.iter().zip(fourier) {
        println!("Fourier\t32\t{name}\t{error:.17e}");
    }
    for (n, method_errors) in [(32, centered_32), (64, centered_64)] {
        for (name, error) in NAMES.iter().zip(method_errors) {
            println!("centered\t{n}\t{name}\t{error:.17e}");
        }
    }
    println!("\nfinite-difference error ratios, n=32 / n=64");
    for ((name, coarse), fine) in NAMES.iter().zip(centered_32).zip(centered_64) {
        println!("{name}\t{:.17e}", coarse / fine);
    }

    assert!(fourier.iter().all(|error| *error < 1e-11));
    assert!(
        centered_32
            .iter()
            .zip(centered_64)
            .all(|(coarse, fine)| (3.5..4.5).contains(&(coarse / fine)))
    );
}
