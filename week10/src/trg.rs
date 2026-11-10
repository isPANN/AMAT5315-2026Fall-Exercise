use nalgebra::DMatrix;
use serde_json::{json, Value};

pub struct ResultData {
    pub log_z: f64,
    pub steps: Vec<Value>,
    pub tensor: Vec<f64>,
    pub dimension: usize,
    pub closure: f64,
}

fn index(d: usize, a: usize, b: usize, c: usize, e: usize) -> usize {
    ((a * d + b) * d + c) * d + e
}
fn split(m: DMatrix<f64>, d: usize, chi: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
    let svd = m.svd(true, true);
    let n = chi.min(svd.singular_values.len());
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let mut left = vec![0.0; d * d * n];
    let mut right = vec![0.0; n * d * d];
    for a in 0..d {
        for b in 0..d {
            for i in 0..n {
                let root = svd.singular_values[i].sqrt();
                left[(a * d + b) * n + i] = u[(a * d + b, i)] * root;
                right[(i * d + a) * d + b] = root * vt[(i, a * d + b)];
            }
        }
    }
    (
        left,
        right,
        svd.singular_values.iter().copied().collect(),
        n,
    )
}

pub fn solve(l: usize, k: f64, chi: usize) -> ResultData {
    let w = [
        [k.cosh().sqrt(), k.sinh().sqrt()],
        [k.cosh().sqrt(), -k.sinh().sqrt()],
    ];
    let mut d = 2;
    let mut t = vec![0.0; 16];
    for a in 0..d {
        for b in 0..d {
            for c in 0..d {
                for e in 0..d {
                    t[index(d, a, b, c, e)] = w.iter().map(|s| s[a] * s[b] * s[c] * s[e]).sum();
                }
            }
        }
    }
    let nsteps = 2 * l.ilog2();
    let mut log_z = 0.0;
    let mut steps = Vec::new();
    for step in 0..nsteps {
        let scale = t.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        assert!(scale > 0.0 && scale.is_finite(), "invalid TRG scale");
        for value in &mut t {
            *value /= scale;
        }
        let count = (l * l) >> step;
        log_z += (count as f64) * scale.ln();
        let ma = DMatrix::from_fn(d * d, d * d, |i, j| t[index(d, j / d, i % d, i / d, j % d)]);
        let mb = DMatrix::from_fn(d * d, d * d, |i, j| t[index(d, j % d, j / d, i % d, i / d)]);
        let (s1, s3, sa, n) = split(ma, d, chi);
        let (s2, s4, sb, n2) = split(mb, d, chi);
        assert_eq!(n, n2);
        let p = DMatrix::from_fn(n * n, d * d, |row, col| {
            let (r, u, w, b) = (row / n, row % n, col / d, col % d);
            (0..d)
                .map(|a| s1[(w * d + a) * n + r] * s2[(a * d + b) * n + u])
                .sum::<f64>()
        });
        let q = DMatrix::from_fn(d * d, n * n, |row, col| {
            let (w, b, left, down) = (row / d, row % d, col / n, col % n);
            (0..d)
                .map(|g| s3[(left * d + b) * d + g] * s4[(down * d + g) * d + w])
                .sum::<f64>()
        });
        let result = p * q;
        let mut next = vec![0.0; n * n * n * n];
        for r in 0..n {
            for u in 0..n {
                for left in 0..n {
                    for down in 0..n {
                        next[index(n, r, u, left, down)] = result[(r * n + u, left * n + down)];
                    }
                }
            }
        }
        steps.push(json!({"scale":scale,"tensor_count":count,"input_dimension":d,"retained_dimension":n,"singular_values_a":sa,"singular_values_b":sb,"s1":s1,"s2":s2,"s3":s3,"s4":s4}));
        d = n;
        t = next;
    }
    let closure = (0..d)
        .flat_map(|i| (0..d).map(move |j| (i, j)))
        .map(|(i, j)| t[index(d, i, j, i, j)])
        .sum::<f64>();
    assert!(closure > 0.0 && closure.is_finite(), "invalid TRG closure");
    log_z += closure.ln();
    ResultData {
        log_z,
        steps,
        tensor: t,
        dimension: d,
        closure,
    }
}

#[cfg(test)]
mod tests {
    use super::solve;

    #[test]
    fn periodic_two_by_two_matches_spin_enumeration() {
        for k in [0.0_f64, 0.3, 0.7] {
            let mut z = 0.0;
            for state in 0usize..16 {
                let spin = |i| {
                    if state & (1usize << i) == 0usize {
                        -1.0
                    } else {
                        1.0
                    }
                };
                let mut bonds = 0.0;
                for y in 0..2 {
                    for x in 0..2 {
                        bonds += spin(y * 2 + x) * spin(y * 2 + (x + 1) % 2);
                        bonds += spin(y * 2 + x) * spin(((y + 1) % 2) * 2 + x);
                    }
                }
                z += (k * bonds).exp();
            }
            assert!((solve(2, k, 16).log_z - z.ln()).abs() < 1e-10);
        }
    }
}
