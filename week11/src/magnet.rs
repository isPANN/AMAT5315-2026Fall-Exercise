use num_bigint::BigUint;
use omeco::{
    contraction_complexity, optimize_code, uniform_size_dict, EinCode, GreedyMethod, NestedEinsum,
    TreeSA,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{json, Value};
use std::{collections::BTreeMap, error::Error, fs, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Magnet {
    fields: Vec<i64>,
    edges: Vec<(usize, usize, i64)>,
}
impl Magnet {
    fn energy(&self, spins: &[i64]) -> i64 {
        self.fields
            .iter()
            .zip(spins)
            .map(|(h, s)| h * s)
            .sum::<i64>()
            + self
                .edges
                .iter()
                .map(|(a, b, j)| j * spins[*a] * spins[*b])
                .sum::<i64>()
    }
}

struct Tensor {
    labels: Vec<usize>,
    logs: Vec<f64>,
    scale: f64,
    mins: Vec<(i64, BigUint)>,
    children: Option<(Box<Tensor>, Box<Tensor>)>,
}
pub(crate) fn index(labels: &[usize], union: &[usize], bits: usize) -> usize {
    labels.iter().enumerate().fold(0, |index, (i, label)| {
        index | (((bits >> union.iter().position(|x| x == label).unwrap()) & 1) << i)
    })
}
pub(crate) fn union(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = a.to_vec();
    for x in b {
        if !result.contains(x) {
            result.push(*x);
        }
    }
    result
}
fn factor(labels: Vec<usize>, energies: Vec<i64>, beta: f64) -> Tensor {
    let scale = energies
        .iter()
        .map(|e| -beta * *e as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    Tensor {
        labels,
        logs: energies.iter().map(|e| -beta * *e as f64 - scale).collect(),
        scale,
        mins: energies
            .into_iter()
            .map(|e| (e, BigUint::from(1u8)))
            .collect(),
        children: None,
    }
}
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        b
    } else if b == f64::NEG_INFINITY {
        a
    } else {
        let m = a.max(b);
        m + ((a - m).exp() + (b - m).exp()).ln()
    }
}
fn contract(tree: &NestedEinsum<usize>, leaves: &[Tensor]) -> Tensor {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            let t = &leaves[*tensor_index];
            Tensor {
                labels: t.labels.clone(),
                logs: t.logs.clone(),
                scale: t.scale,
                mins: t.mins.clone(),
                children: None,
            }
        }
        NestedEinsum::Node { args, eins } => {
            assert_eq!(args.len(), 2);
            let left = Box::new(contract(&args[0], leaves));
            let right = Box::new(contract(&args[1], leaves));
            let all = union(&left.labels, &right.labels);
            let labels = eins.iy.clone();
            let mut logs = vec![f64::NEG_INFINITY; 1usize << labels.len()];
            let mut mins = vec![(i64::MAX, BigUint::from(0u8)); logs.len()];
            for bits in 0..(1usize << all.len()) {
                let a = index(&left.labels, &all, bits);
                let b = index(&right.labels, &all, bits);
                let c = index(&labels, &all, bits);
                logs[c] = log_add(logs[c], left.logs[a] + right.logs[b]);
                let energy = left.mins[a].0 + right.mins[b].0;
                let count = &left.mins[a].1 * &right.mins[b].1;
                if energy < mins[c].0 {
                    mins[c] = (energy, count);
                } else if energy == mins[c].0 {
                    mins[c].1 += count;
                }
            }
            let shift = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for x in &mut logs {
                *x -= shift;
            }
            Tensor {
                labels,
                logs,
                scale: left.scale + right.scale + shift,
                mins,
                children: Some((left, right)),
            }
        }
    }
}
fn sample(node: &Tensor, assignment: &mut [Option<usize>], rng: &mut StdRng) {
    if let Some((left, right)) = &node.children {
        let labels = union(&left.labels, &right.labels);
        let unknown: Vec<_> = labels
            .iter()
            .filter(|l| assignment[**l].is_none())
            .copied()
            .collect();
        let mut options = Vec::new();
        for bits in 0..(1usize << unknown.len()) {
            for (i, var) in unknown.iter().enumerate() {
                assignment[*var] = Some((bits >> i) & 1);
            }
            let get_index = |ls: &[usize]| {
                ls.iter()
                    .enumerate()
                    .fold(0, |v, (i, l)| v | (assignment[*l].unwrap() << i))
            };
            options.push(left.logs[get_index(&left.labels)] + right.logs[get_index(&right.labels)]);
        }
        let max = options.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = options.iter().map(|x| (x - max).exp()).collect();
        let mut pick = rng.random::<f64>() * weights.iter().sum::<f64>();
        let mut selected = weights.len() - 1;
        for (i, weight) in weights.iter().enumerate() {
            pick -= weight;
            if pick < 0.0 {
                selected = i;
                break;
            }
        }
        for (i, var) in unknown.iter().enumerate() {
            assignment[*var] = Some((selected >> i) & 1);
        }
        sample(left, assignment, rng);
        sample(right, assignment, rng);
    }
}

pub fn run(args: &[String]) -> Result<()> {
    let start = Instant::now();
    let mut options = BTreeMap::new();
    for pair in args.chunks(2) {
        if pair.len() != 2 {
            return Err("each option requires a value".into());
        }
        match pair[0].as_str() {
            "--mode" | "--seed" | "--out" | "--l" | "--p-af" | "--beta" | "--input" | "--order"
            | "--samples" | "--n" => (),
            _ => return Err(format!("unknown option: {}", pair[0]).into()),
        }
        if options.insert(pair[0].as_str(), pair[1].as_str()).is_some() {
            return Err(format!("repeated option: {}", pair[0]).into());
        }
    }
    let required = |name| {
        options
            .get(name)
            .copied()
            .ok_or_else(|| format!("missing {name}"))
    };
    let mode = required("--mode")?;
    if !["plan", "count", "sample"].contains(&mode) {
        return Err(format!("unsupported mode: {mode}").into());
    }
    let seed: u64 = required("--seed")?.parse()?;
    let out = required("--out")?;
    let order = options.get("--order").copied().unwrap_or("treesa");
    if !["greedy", "treesa"].contains(&order) {
        return Err("--order must be greedy or treesa".into());
    }
    if options.contains_key("--n") {
        return crate::nqueens::run(&options, start);
    }
    let beta = if mode == "plan" {
        if options.contains_key("--beta") || options.contains_key("--samples") {
            return Err("--beta and --samples are not used in plan mode".into());
        }
        None
    } else {
        let b: f64 = required("--beta")?.parse()?;
        if !b.is_finite() {
            return Err("--beta must be finite".into());
        }
        Some(b)
    };
    let samples = if mode == "sample" {
        Some(required("--samples")?.parse::<usize>()?)
    } else {
        if options.contains_key("--samples") {
            return Err("--samples is only used in sample mode".into());
        }
        None
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let magnet = if let Some(path) = options.get("--input") {
        if options.contains_key("--l") || options.contains_key("--p-af") {
            return Err("--input cannot be combined with --l or --p-af".into());
        }
        let bundle: Value = serde_json::from_slice(&fs::read(path)?)?;
        if bundle["target"]["type"] != "SpinGlass" {
            return Err("bundle target must be SpinGlass".into());
        }
        let data = &bundle["target"]["data"];
        let n = data["graph"]["num_vertices"]
            .as_u64()
            .ok_or("missing num_vertices")? as usize;
        let fields: Vec<i64> = serde_json::from_value(data["fields"].clone())?;
        let pairs: Vec<[usize; 2]> = serde_json::from_value(data["graph"]["edges"].clone())?;
        let couplings: Vec<i64> = serde_json::from_value(data["couplings"].clone())?;
        if n == 0
            || fields.len() != n
            || pairs.len() != couplings.len()
            || pairs.iter().any(|[a, b]| *a >= n || *b >= n || a == b)
        {
            return Err("invalid SpinGlass dimensions or edges".into());
        }
        Magnet {
            fields,
            edges: pairs
                .into_iter()
                .zip(couplings)
                .map(|([a, b], j)| (a, b, j))
                .collect(),
        }
    } else {
        let l: usize = required("--l")?.parse()?;
        let p: f64 = required("--p-af")?.parse()?;
        if l == 0 || !(0.0..=1.0).contains(&p) {
            return Err("--l must be positive and --p-af must be between 0 and 1".into());
        }
        let n = l.checked_mul(l).ok_or("lattice size overflow")?;
        let mut edges = Vec::new();
        for row in 0..l {
            for col in 0..l {
                let i = row * l + col;
                if col + 1 < l {
                    edges.push((i, i + 1, if rng.random::<f64>() < p { 1 } else { -1 }));
                }
                if row + 1 < l {
                    edges.push((i, i + l, if rng.random::<f64>() < p { 1 } else { -1 }));
                }
            }
        }
        Magnet {
            fields: vec![0; n],
            edges,
        }
    };
    let mut leaves = Vec::new();
    let b = beta.unwrap_or(0.0);
    for (i, &h) in magnet.fields.iter().enumerate() {
        leaves.push(factor(vec![i], vec![-h, h], b));
    }
    for &(a, c, j) in &magnet.edges {
        leaves.push(factor(vec![a, c], vec![j, -j, -j, j], b));
    }
    leaves.push(factor(vec![], vec![0], b));
    let code = EinCode::new(leaves.iter().map(|t| t.labels.clone()).collect(), vec![]);
    let sizes = uniform_size_dict(&code, 2);
    let tree = match order {
        "greedy" => optimize_code(&code, &sizes, &GreedyMethod::default()),
        "treesa" => optimize_code(&code, &sizes, &TreeSA::default()),
        _ => unreachable!(),
    }
    .ok_or("omeco failed to produce a contraction tree")?;
    let complexity = contraction_complexity(&tree, &sizes, &code.ixs);
    let mut summary =
        json!({"variables": magnet.fields.len(), "tc": complexity.tc, "sc": complexity.sc});
    let mut evidence =
        json!({"fields": magnet.fields, "edges": magnet.edges, "order": order, "tree": tree});
    if beta.is_some() {
        let root = contract(&tree, &leaves);
        let log_z = root.scale + root.logs[0];
        if !log_z.is_finite() {
            return Err(
                "non-finite partition function; beta or energies exceed numerical range".into(),
            );
        }
        summary["log_z"] = json!(log_z);
        summary["e_min"] = json!(root.mins[0].0);
        summary["g_0"] = json!(root.mins[0].1.to_string());
        evidence["root_log_scale"] = json!(root.scale);
        evidence["root_log_value"] = json!(root.logs[0]);
        if let Some(count) = samples {
            let mut draws = Vec::with_capacity(count);
            for _ in 0..count {
                let mut assignment = vec![None; magnet.fields.len()];
                sample(&root, &mut assignment, &mut rng);
                let spins: Vec<i64> = assignment
                    .into_iter()
                    .map(|x| 2 * x.expect("tree must assign every spin") as i64 - 1)
                    .collect();
                draws.push(json!({"energy": magnet.energy(&spins), "spins": spins}));
            }
            summary["samples"] = json!(count);
            evidence["samples"] = json!(draws);
        }
    }
    let runtime_ms = start.elapsed().as_secs_f64() * 1000.0;
    summary["runtime_ms"] = json!(runtime_ms);
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
                "{mode}\t{key}\t{}\t{runtime_ms}",
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string())
            );
        }
    }
    Ok(())
}
