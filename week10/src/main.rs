use nalgebra::{linalg::SVD, DMatrix, Dyn};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use std::{
    fs::{self, File},
    io::{self, BufWriter, Write},
};

fn reconstruct(svd: &SVD<f64, Dyn, Dyn>, rank: usize) -> DMatrix<f64> {
    let u = svd.u.as_ref().unwrap().columns(0, rank);
    let vt = svd.v_t.as_ref().unwrap().rows(0, rank);
    u * DMatrix::from_diagonal(&svd.singular_values.rows(0, rank)) * vt
}

fn write_values(path: &str, values: impl Iterator<Item = f64>) -> io::Result<()> {
    let mut output = BufWriter::new(File::create(path)?);
    for value in values {
        output.write_all(&value.to_le_bytes())?;
    }
    output.flush()
}

fn write_matrix(path: &str, matrix: &DMatrix<f64>) -> io::Result<()> {
    write_values(
        path,
        (0..matrix.nrows()).flat_map(|i| (0..matrix.ncols()).map(move |j| matrix[(i, j)])),
    )
}

fn main() -> io::Result<()> {
    fs::create_dir_all("results")?;
    let mut rng = ChaCha20Rng::seed_from_u64(2026);
    let matrix = DMatrix::from_fn(512, 512, |_, _| StandardNormal.sample(&mut rng));
    write_matrix("results/matrix.bin", &matrix)?;
    let svd = matrix.svd(true, true);
    write_values(
        "results/singular-values.bin",
        svd.singular_values.iter().copied(),
    )?;
    for rank in [16, 64, 128, 256, 512] {
        write_matrix(
            &format!("results/rank-{rank}.bin"),
            &reconstruct(&svd, rank),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::reconstruct;
    use nalgebra::DMatrix;

    #[test]
    fn truncated_reconstruction_keeps_largest_singular_values() {
        let matrix = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![3.0, 2.0, 1.0]));
        let svd = matrix.svd(true, true);
        let rank_one = reconstruct(&svd, 1);
        let expected = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![3.0, 0.0, 0.0]));
        assert!((rank_one - expected).norm() < 1e-12);
        let full = reconstruct(&svd, 3);
        let original = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![3.0, 2.0, 1.0]));
        assert!((full - original).norm() < 1e-12);
    }
}
