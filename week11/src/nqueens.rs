use crate::magnet::union;
use num_bigint::BigUint;
use omeco::{
    contraction_complexity, optimize_code, uniform_size_dict, EinCode, GreedyMethod, NestedEinsum,
    TreeSA,
};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use serde_json::json;
use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    fs,
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Tensor {
    labels: Vec<usize>,
    values: HashMap<usize, BigUint>,
    children: Option<(Box<Tensor>, Box<Tensor>)>,
}

fn add_line(leaves: &mut Vec<Tensor>, next: &mut usize, squares: &[usize], required: bool) {
    let mut carry = None;
    for (position, &square) in squares.iter().enumerate() {
        let last = position + 1 == squares.len();
        let outgoing = if last {
            None
        } else {
            let id = *next;
            *next += 1;
            Some(id)
        };
        let mut labels = vec![square];
        if let Some(id) = carry {
            labels.push(id);
        }
        if let Some(id) = outgoing {
            labels.push(id);
        }
        let mut values = HashMap::new();
        for bits in 0..(1usize << labels.len()) {
            let x = bits & 1;
            let incoming = carry
                .map(|id| (bits >> labels.iter().position(|v| *v == id).unwrap()) & 1)
                .unwrap_or(0);
            let sum = x + incoming;
            let valid = sum <= 1
                && if let Some(id) = outgoing {
                    sum == ((bits >> labels.iter().position(|v| *v == id).unwrap()) & 1)
                } else {
                    !required || sum == 1
                };
            if valid {
                values.insert(bits, BigUint::from(1u8));
            }
        }
        leaves.push(Tensor {
            labels,
            values,
            children: None,
        });
        carry = outgoing;
    }
}

fn projection(source: &[usize], target: &[usize]) -> Vec<(usize, usize)> {
    target
        .iter()
        .enumerate()
        .filter_map(|(to, label)| {
            source
                .iter()
                .position(|x| x == label)
                .map(|from| (from, to))
        })
        .collect()
}

fn project(bits: usize, mapping: &[(usize, usize)]) -> usize {
    mapping.iter().fold(0, |result, (from, to)| {
        result | (((bits >> from) & 1) << to)
    })
}

fn contract(tree: &NestedEinsum<usize>, leaves: &[Tensor]) -> Tensor {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            let t = &leaves[*tensor_index];
            Tensor {
                labels: t.labels.clone(),
                values: t.values.clone(),
                children: None,
            }
        }
        NestedEinsum::Node { args, eins } => {
            assert_eq!(args.len(), 2);
            let left = Box::new(contract(&args[0], leaves));
            let right = Box::new(contract(&args[1], leaves));
            let labels = eins.iy.clone();
            let common: Vec<_> = left
                .labels
                .iter()
                .filter(|x| right.labels.contains(x))
                .copied()
                .collect();
            let left_key = projection(&left.labels, &common);
            let right_key = projection(&right.labels, &common);
            let left_output = projection(&left.labels, &labels);
            let right_output = projection(&right.labels, &labels);
            let mut grouped: HashMap<usize, Vec<(usize, &BigUint)>> = HashMap::new();
            for (&bits, value) in &right.values {
                grouped
                    .entry(project(bits, &right_key))
                    .or_default()
                    .push((project(bits, &right_output), value));
            }
            let mut values = HashMap::new();
            for (&bits, value) in &left.values {
                if let Some(matches) = grouped.get(&project(bits, &left_key)) {
                    let output = project(bits, &left_output);
                    for (other_output, other_value) in matches {
                        *values.entry(output | other_output).or_default() += value * *other_value;
                    }
                }
            }
            Tensor {
                labels,
                values,
                children: Some((left, right)),
            }
        }
    }
}

fn random_below(limit: &BigUint, rng: &mut StdRng) -> BigUint {
    let bits = limit.bits();
    let mut bytes = vec![0u8; bits.div_ceil(8) as usize];
    loop {
        rng.fill_bytes(&mut bytes);
        if !bits.is_multiple_of(8) {
            *bytes.last_mut().unwrap() &= (1u8 << (bits % 8)) - 1;
        }
        let value = BigUint::from_bytes_le(&bytes);
        if &value < limit {
            return value;
        }
    }
}

fn sample(node: &Tensor, assignment: &mut [Option<usize>], rng: &mut StdRng) {
    if let Some((left, right)) = &node.children {
        let all = union(&left.labels, &right.labels);
        let unknown: Vec<_> = all
            .iter()
            .filter(|x| assignment[**x].is_none())
            .copied()
            .collect();
        let mut weights = Vec::new();
        for bits in 0..(1usize << unknown.len()) {
            for (i, &var) in unknown.iter().enumerate() {
                assignment[var] = Some((bits >> i) & 1);
            }
            let get = |labels: &[usize]| {
                labels
                    .iter()
                    .enumerate()
                    .fold(0, |v, (i, label)| v | (assignment[*label].unwrap() << i))
            };
            weights.push(
                match (
                    left.values.get(&get(&left.labels)),
                    right.values.get(&get(&right.labels)),
                ) {
                    (Some(a), Some(b)) => a * b,
                    _ => BigUint::from(0u8),
                },
            );
        }
        let total: BigUint = weights.iter().sum();
        let mut pick = random_below(&total, rng);
        let selected = weights
            .iter()
            .position(|weight| {
                if &pick < weight {
                    true
                } else {
                    pick -= weight;
                    false
                }
            })
            .unwrap();
        for (i, &var) in unknown.iter().enumerate() {
            assignment[var] = Some((selected >> i) & 1);
        }
        sample(left, assignment, rng);
        sample(right, assignment, rng);
    }
}

pub fn run(options: &BTreeMap<&str, &str>, start: Instant) -> Result<()> {
    let mode = options["--mode"];
    let n: usize = options["--n"].parse()?;
    if n == 0 {
        return Err("--n must be positive".into());
    }
    for name in ["--l", "--p-af", "--input", "--beta"] {
        if options.contains_key(name) {
            return Err(format!("{name} cannot be used with --n").into());
        }
    }
    let draws = if mode == "sample" {
        Some(
            options
                .get("--samples")
                .ok_or("missing --samples")?
                .parse::<usize>()?,
        )
    } else {
        if options.contains_key("--samples") {
            return Err("--samples is only used in sample mode".into());
        }
        None
    };
    let order = options.get("--order").copied().unwrap_or("treesa");
    let seed: u64 = options["--seed"].parse()?;
    let out = options["--out"];
    let squares = n.checked_mul(n).ok_or("board size overflow")?;
    let mut next = squares;
    let mut leaves = Vec::new();
    for row in 0..n {
        add_line(
            &mut leaves,
            &mut next,
            &(0..n).map(|col| row * n + col).collect::<Vec<_>>(),
            true,
        );
    }
    for col in 0..n {
        add_line(
            &mut leaves,
            &mut next,
            &(0..n).map(|row| row * n + col).collect::<Vec<_>>(),
            true,
        );
    }
    for direction in [-1isize, 1] {
        for col in 0..n {
            let mut line = Vec::new();
            let mut r = 0usize;
            let mut c = col as isize;
            while r < n && c >= 0 && c < n as isize {
                line.push(r * n + c as usize);
                r += 1;
                c += direction;
            }
            add_line(&mut leaves, &mut next, &line, false);
        }
        for row in 1..n {
            let mut line = Vec::new();
            let mut r = row;
            let mut c = if direction == 1 { 0 } else { n as isize - 1 };
            while r < n && c >= 0 && c < n as isize {
                line.push(r * n + c as usize);
                r += 1;
                c += direction;
            }
            add_line(&mut leaves, &mut next, &line, false);
        }
    }
    let code = EinCode::new(leaves.iter().map(|t| t.labels.clone()).collect(), vec![]);
    let sizes = uniform_size_dict(&code, 2);
    let tree = match order {
        "greedy" => optimize_code(&code, &sizes, &GreedyMethod::default()),
        "treesa" => optimize_code(&code, &sizes, &TreeSA::default()),
        _ => unreachable!(),
    }
    .ok_or("omeco failed to produce a contraction tree")?;
    let complexity = contraction_complexity(&tree, &sizes, &code.ixs);
    let mut summary = json!({"variables": next, "tc": complexity.tc, "sc": complexity.sc});
    let mut evidence = json!({"n": n, "order": order, "tree": tree});
    if mode != "plan" {
        let root = contract(&tree, &leaves);
        let count = root.values.get(&0).cloned().unwrap_or_default();
        summary["count"] = json!(count.to_string());
        if let Some(draws) = draws {
            if count == BigUint::from(0u8) && draws > 0 {
                return Err("cannot sample: no valid N-queens boards".into());
            }
            let mut rng = StdRng::seed_from_u64(seed);
            let mut boards = Vec::with_capacity(draws);
            for _ in 0..draws {
                let mut assignment = vec![None; next];
                sample(&root, &mut assignment, &mut rng);
                let board: Vec<Vec<usize>> = (0..n)
                    .map(|r| {
                        (0..n)
                            .map(|c| assignment[r * n + c].expect("tree must assign every square"))
                            .collect()
                    })
                    .collect();
                boards.push(json!({"board": board}));
            }
            evidence["samples"] = json!(boards);
            summary["samples"] = json!(draws);
        }
    }
    let runtime = start.elapsed().as_secs_f64() * 1000.0;
    summary["runtime_ms"] = json!(runtime);
    let run = json!({"schema_version": 1, "mode": mode, "seed": seed, "parameters": options, "summary": summary, "evidence": evidence});
    fs::create_dir_all(out)?;
    fs::write(
        format!("{out}/run.json"),
        serde_json::to_string_pretty(&run)? + "\n",
    )?;
    println!("mode\tmetric\tvalue\truntime_ms");
    for (key, value) in run["summary"].as_object().unwrap() {
        if key != "runtime_ms" {
            println!(
                "{mode}\t{key}\t{}\t{runtime}",
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string())
            );
        }
    }
    Ok(())
}
