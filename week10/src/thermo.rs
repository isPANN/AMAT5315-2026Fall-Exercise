use nalgebra::DMatrix;
use serde_json::json;
use std::{env, fs, time::Instant};

#[derive(Clone)]
struct Tensor {
    left: usize,
    right: usize,
    data: Vec<f64>,
}
impl Tensor {
    fn new(left: usize, right: usize) -> Self {
        Self {
            left,
            right,
            data: vec![0.0; left * 2 * right],
        }
    }
    fn get(&self, l: usize, p: usize, r: usize) -> f64 {
        self.data[(l * 2 + p) * self.right + r]
    }
    fn set(&mut self, l: usize, p: usize, r: usize, v: f64) {
        self.data[(l * 2 + p) * self.right + r] = v;
    }
}
struct ResultData {
    log_z: f64,
    tensors: Vec<Tensor>,
    scales: Vec<f64>,
    discarded: Vec<Vec<f64>>,
    final_contraction: f64,
}

fn solve(l: usize, k: f64, chi: usize) -> ResultData {
    let c = k.cosh().sqrt();
    let s = k.sinh().sqrt();
    let w = [[c, s], [c, -s]];
    let b = 1.0 / c;
    let mut mps = vec![
        Tensor {
            left: 1,
            right: 1,
            data: vec![b, 0.0]
        };
        l
    ];
    let mut scales = Vec::new();
    let mut discarded = Vec::new();
    for _ in 0..l {
        let mut next = Vec::new();
        for (x, old) in mps.iter().enumerate() {
            let hl = if x == 0 { 1 } else { 2 };
            let hr = if x == l - 1 { 1 } else { 2 };
            let mut t = Tensor::new(old.left * hl, old.right * hr);
            for ol in 0..old.left {
                for a in 0..hl {
                    for p in 0..2 {
                        for or in 0..old.right {
                            for d in 0..hr {
                                let mut value = 0.0;
                                for q in 0..2 {
                                    for ws in &w {
                                        value += old.get(ol, q, or) * ws[a] * ws[p] * ws[d] * ws[q];
                                    }
                                }
                                if x == 0 {
                                    value *= b;
                                }
                                if x == l - 1 {
                                    value *= b;
                                }
                                t.set(ol * hl + a, p, or * hr + d, value);
                            }
                        }
                    }
                }
            }
            next.push(t);
        }
        mps = next;
        // Right-canonicalize before truncating left to right.
        for x in (1..l).rev() {
            let t = &mps[x];
            let mat = DMatrix::from_fn(t.left, 2 * t.right, |i, j| {
                t.get(i, j / t.right, j % t.right)
            });
            let svd = mat.svd(true, true);
            let n = svd.singular_values.len();
            let transfer = svd.u.unwrap() * DMatrix::from_diagonal(&svd.singular_values);
            let vt = svd.v_t.unwrap();
            let mut right = Tensor::new(n, t.right);
            for i in 0..n {
                for p in 0..2 {
                    for j in 0..t.right {
                        right.set(i, p, j, vt[(i, p * t.right + j)]);
                    }
                }
            }
            let old = &mps[x - 1];
            let mut left = Tensor::new(old.left, n);
            for i in 0..old.left {
                for p in 0..2 {
                    for j in 0..n {
                        left.set(
                            i,
                            p,
                            j,
                            (0..old.right)
                                .map(|a| old.get(i, p, a) * transfer[(a, j)])
                                .sum(),
                        );
                    }
                }
            }
            mps[x - 1] = left;
            mps[x] = right;
        }
        let mut row_discarded = Vec::new();
        for x in 0..l - 1 {
            let t = &mps[x];
            let mat = DMatrix::from_fn(t.left * 2, t.right, |i, j| t.get(i / 2, i % 2, j));
            let svd = mat.svd(true, true);
            let n = chi.min(svd.singular_values.len());
            row_discarded.push(
                svd.singular_values
                    .iter()
                    .skip(n)
                    .map(|v| v * v)
                    .sum::<f64>(),
            );
            let u = svd.u.unwrap();
            let vt = svd.v_t.unwrap();
            let mut left = Tensor::new(t.left, n);
            for i in 0..t.left {
                for p in 0..2 {
                    for j in 0..n {
                        left.set(i, p, j, u[(i * 2 + p, j)]);
                    }
                }
            }
            let old = &mps[x + 1];
            let mut right = Tensor::new(n, old.right);
            for i in 0..n {
                for p in 0..2 {
                    for j in 0..old.right {
                        right.set(
                            i,
                            p,
                            j,
                            svd.singular_values[i]
                                * (0..old.left)
                                    .map(|a| vt[(i, a)] * old.get(a, p, j))
                                    .sum::<f64>(),
                        );
                    }
                }
            }
            mps[x] = left;
            mps[x + 1] = right;
        }
        let last = &mut mps[l - 1];
        let scale = last.data.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            scale.is_finite() && scale > 0.0,
            "invalid normalization scale"
        );
        for v in &mut last.data {
            *v /= scale;
        }
        scales.push(scale);
        discarded.push(row_discarded);
    }
    let mut v = vec![1.0];
    for t in &mps {
        let mut next = vec![0.0; t.right];
        for i in 0..t.left {
            for j in 0..t.right {
                next[j] += v[i] * t.get(i, 0, j) * b;
            }
        }
        v = next;
    }
    let final_contraction = v[0];
    assert!(
        final_contraction > 0.0 && final_contraction.is_finite(),
        "invalid final contraction"
    );
    let log_z = scales.iter().map(|x| x.ln()).sum::<f64>() + final_contraction.ln();
    ResultData {
        log_z,
        tensors: mps,
        scales,
        discarded,
        final_contraction,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 10 {
        return Err("usage: thermo --method boundary --l L --k K --chi CHI --out DIR".into());
    }
    let mut method = None;
    let mut l = None;
    let mut k = None;
    let mut chi = None;
    let mut out = None;
    for pair in args.chunks_exact(2) {
        match pair[0].as_str() {
            "--method" => method = Some(pair[1].as_str()),
            "--l" => l = Some(pair[1].parse::<usize>()?),
            "--k" => k = Some(pair[1].parse::<f64>()?),
            "--chi" => chi = Some(pair[1].parse::<usize>()?),
            "--out" => out = Some(pair[1].as_str()),
            _ => return Err("unknown argument".into()),
        }
    }
    let (Some("boundary"), Some(l), Some(k), Some(chi), Some(out)) = (method, l, k, chi, out)
    else {
        return Err("all arguments are required; method must be boundary".into());
    };
    if l < 2 || !k.is_finite() || k < 0.0 || chi < 1 {
        return Err("require L >= 2, finite K >= 0, and chi >= 1".into());
    }
    let start = Instant::now();
    let result = solve(l, k, chi);
    let seconds = start.elapsed().as_secs_f64();
    let tensors: Vec<_> = result
        .tensors
        .iter()
        .map(|t| json!({"shape":[t.left,2,t.right],"data":t.data}))
        .collect();
    let evidence = json!({"model":"zero-field Ising, H=-sum_edges s_i s_j, spins ±1", "geometry":"open square", "method":"boundary MPS, right-canonical then left-to-right SVD", "l":l,"k":k,"chi":chi,"log_z":result.log_z,"g":result.log_z/(l*l) as f64,"solver_seconds":seconds,"normalization_scales":result.scales,"discarded_squared_singular_values":result.discarded,"final_contraction":result.final_contraction,"missing_bond_vector":[1.0/k.cosh().sqrt(),0.0],"tensor_index_order":"left, physical, right; row-major", "final_tensors":tensors});
    fs::create_dir_all(out)?;
    fs::write(
        format!("{out}/evidence.json"),
        serde_json::to_vec_pretty(&evidence)?,
    )?;
    println!(
        "log Z = {:.15}, g = {:.15}",
        result.log_z,
        result.log_z / (l * l) as f64
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::solve;
    #[test]
    fn open_square_matches_enumeration_without_truncation() {
        for l in [2, 3] {
            for k in [0.0_f64, 0.3, 0.7] {
                let mut z: f64 = 0.0;
                for state in 0..(1usize << (l * l)) {
                    let spin = |i| {
                        if state & (1usize << i) == 0 {
                            -1.0
                        } else {
                            1.0
                        }
                    };
                    let mut bonds = 0.0;
                    for y in 0..l {
                        for x in 0..l {
                            if x + 1 < l {
                                bonds += spin(y * l + x) * spin(y * l + x + 1);
                            }
                            if y + 1 < l {
                                bonds += spin(y * l + x) * spin((y + 1) * l + x);
                            }
                        }
                    }
                    z += (k * bonds).exp();
                }
                let result = solve(l, k, 32);
                assert!((result.log_z - z.ln()).abs() < 1e-10);
            }
        }
    }
}
