use serde::Deserialize;
use serde_json::{Value, json};
use std::env;
use std::error::Error;
use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Experiment {
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
    steps: usize,
    source_frequency: f64,
    source_peak_time: f64,
    source_amplitude: f64,
    sponge_width: usize,
    sponge_strength: f64,
    shots: Vec<[f64; 2]>,
    receivers: Vec<[usize; 2]>,
    background: Vec<Vec<f64>>,
    perturbation: Option<Vec<Vec<f64>>>,
}

unsafe extern "C" {
    fn seismic_step(
        previous: *const f64,
        current: *const f64,
        speed: *const f64,
        damping_dt: *const f64,
        footprint: *const f64,
        next: *mut f64,
        nx: usize,
        nz: usize,
        inv_dx_sq: f64,
        dt_sq: f64,
        pulse: f64,
    );
    fn seismic_step_jvp(
        previous: *const f64,
        dprevious: *const f64,
        current: *const f64,
        dcurrent: *const f64,
        speed: *const f64,
        dspeed: *const f64,
        damping_dt: *const f64,
        footprint: *const f64,
        next: *mut f64,
        dnext: *mut f64,
        nx: usize,
        nz: usize,
        inv_dx_sq: f64,
        dt_sq: f64,
        pulse: f64,
    );
    fn seismic_step_vjp(
        previous: *const f64,
        dprevious: *mut f64,
        current: *const f64,
        dcurrent: *mut f64,
        speed: *const f64,
        dspeed: *mut f64,
        damping_dt: *const f64,
        footprint: *const f64,
        next: *mut f64,
        dnext: *mut f64,
        nx: usize,
        nz: usize,
        inv_dx_sq: f64,
        dt_sq: f64,
        pulse: f64,
    );
}

fn advance(
    previous: &[f64],
    current: &[f64],
    speed: &[f64],
    damping_dt: &[f64],
    footprint: &[f64],
    next: &mut [f64],
    tangent: Option<(&[f64], &[f64], &[f64], &mut [f64])>,
    nx: usize,
    nz: usize,
    inv_dx_sq: f64,
    dt_sq: f64,
    pulse: f64,
) {
    let cells = nx * nz;
    for field in [previous, current, speed, damping_dt, footprint, next] {
        assert_eq!(field.len(), cells);
    }
    match tangent {
        Some((dprevious, dcurrent, dspeed, dnext)) => {
            for field in [dprevious, dcurrent, dspeed, &*dnext] {
                assert_eq!(field.len(), cells);
            }
            unsafe {
                seismic_step_jvp(
                    previous.as_ptr(),
                    dprevious.as_ptr(),
                    current.as_ptr(),
                    dcurrent.as_ptr(),
                    speed.as_ptr(),
                    dspeed.as_ptr(),
                    damping_dt.as_ptr(),
                    footprint.as_ptr(),
                    next.as_mut_ptr(),
                    dnext.as_mut_ptr(),
                    nx,
                    nz,
                    inv_dx_sq,
                    dt_sq,
                    pulse,
                );
            }
        }
        None => unsafe {
            seismic_step(
                previous.as_ptr(),
                current.as_ptr(),
                speed.as_ptr(),
                damping_dt.as_ptr(),
                footprint.as_ptr(),
                next.as_mut_ptr(),
                nx,
                nz,
                inv_dx_sq,
                dt_sq,
                pulse,
            );
        },
    }
}

fn reverse_step(
    previous: &[f64],
    dprevious: &mut [f64],
    current: &[f64],
    dcurrent: &mut [f64],
    speed: &[f64],
    dspeed: &mut [f64],
    damping_dt: &[f64],
    footprint: &[f64],
    next: &mut [f64],
    dnext: &mut [f64],
    nx: usize,
    nz: usize,
    inv_dx_sq: f64,
    dt_sq: f64,
    pulse: f64,
) {
    let cells = nx * nz;
    for field in [
        previous,
        &*dprevious,
        current,
        &*dcurrent,
        speed,
        &*dspeed,
        damping_dt,
        footprint,
        &*next,
        &*dnext,
    ] {
        assert_eq!(field.len(), cells);
    }
    unsafe {
        seismic_step_vjp(
            previous.as_ptr(),
            dprevious.as_mut_ptr(),
            current.as_ptr(),
            dcurrent.as_mut_ptr(),
            speed.as_ptr(),
            dspeed.as_mut_ptr(),
            damping_dt.as_ptr(),
            footprint.as_ptr(),
            next.as_mut_ptr(),
            dnext.as_mut_ptr(),
            nx,
            nz,
            inv_dx_sq,
            dt_sq,
            pulse,
        );
    }
}

fn npy_header(writer: &mut impl Write, dtype: &str, shape: &[usize]) -> std::io::Result<()> {
    let dimensions = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let mut header =
        format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': ({dimensions}), }}");
    let padding = (16 - (10 + header.len() + 1) % 16) % 16;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    writer.write_all(b"\x93NUMPY\x01\x00")?;
    writer.write_all(&(header.len() as u16).to_le_bytes())?;
    writer.write_all(header.as_bytes())
}

fn read_born_data(path: &Path, shape: &[usize; 3]) -> Result<Vec<f64>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let mut header = Vec::new();
    npy_header(&mut header, "<f8", shape)?;
    let count = shape.iter().product::<usize>();
    if !bytes.starts_with(&header) || bytes.len() != header.len() + count * 8 {
        return Err(
            "--data must be born_data.npy with matching shot, step, and receiver dimensions".into(),
        );
    }
    let values: Vec<f64> = bytes[header.len()..]
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("--data contains nonfinite receiver weights".into());
    }
    Ok(values)
}

fn ricker(e: &Experiment, step: usize) -> f64 {
    let phase = PI * e.source_frequency * (step as f64 * e.dt - e.source_peak_time);
    let phase_sq = phase * phase;
    e.source_amplitude * (1.0 - 2.0 * phase_sq) * (-phase_sq).exp()
}

fn write_json(path: &Path, value: &Value) -> Result<(), Box<dyn Error>> {
    let mut file = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut file, value)?;
    writeln!(file)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut experiment_path: Option<PathBuf> = None;
    let mut mode = None;
    let mut out: Option<PathBuf> = None;
    let mut data_path: Option<PathBuf> = None;
    let mut recording_every = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--experiment" => experiment_path = Some(value.into()),
            "--mode" => mode = Some(value),
            "--out" => out = Some(value.into()),
            "--data" => data_path = Some(value.into()),
            "--every" => recording_every = Some(value.parse::<usize>()?),
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    let experiment_path = experiment_path.ok_or("missing --experiment")?;
    let mode = mode.ok_or("missing --mode")?;
    let out = out.ok_or("missing --out")?;
    if mode != "forward" && mode != "born" && mode != "adjoint" {
        return Err(format!("mode {mode} is not implemented").into());
    }
    if mode == "adjoint" && data_path.is_none() {
        return Err("adjoint mode requires --data".into());
    }
    if mode != "adjoint" && data_path.is_some() {
        return Err("--data is only valid in adjoint mode".into());
    }
    if recording_every == Some(0) {
        return Err("--every must be positive".into());
    }
    if mode == "born" && recording_every.is_some() {
        return Err("--every is only valid in forward and adjoint modes".into());
    }

    let raw: Value = serde_json::from_slice(&fs::read(&experiment_path)?)?;
    let experiment: Experiment = serde_json::from_value(raw.clone())?;
    let e = &experiment;
    if e.nx < 3
        || e.nz < 3
        || e.steps == 0
        || !e.dx.is_finite()
        || e.dx <= 0.0
        || !e.dt.is_finite()
        || e.dt <= 0.0
    {
        return Err("grid dimensions, spacing, timestep, and steps must be positive".into());
    }
    if e.sponge_width == 0 || !e.sponge_strength.is_finite() || e.sponge_strength < 0.0 {
        return Err("sponge width must be positive and strength nonnegative".into());
    }
    if !e.source_frequency.is_finite()
        || e.source_frequency <= 0.0
        || !e.source_peak_time.is_finite()
        || !e.source_amplitude.is_finite()
    {
        return Err("source parameters must be finite and frequency positive".into());
    }
    if e.background.len() != e.nz || e.background.iter().any(|row| row.len() != e.nx) {
        return Err("background shape does not match nx and nz".into());
    }
    let mut speed = Vec::with_capacity(e.nx * e.nz);
    let mut max_speed: f64 = 0.0;
    for &value in e.background.iter().flatten() {
        if !value.is_finite() || value <= 0.0 {
            return Err("background speeds must be finite and positive".into());
        }
        max_speed = max_speed.max(value);
        speed.push(value);
    }
    let perturbation = if mode == "born" {
        let rows = e
            .perturbation
            .as_ref()
            .ok_or("born mode requires perturbation")?;
        if rows.len() != e.nz || rows.iter().any(|row| row.len() != e.nx) {
            return Err("perturbation shape does not match nx and nz".into());
        }
        let values: Vec<f64> = rows.iter().flatten().copied().collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err("perturbation values must be finite".into());
        }
        values
    } else {
        Vec::new()
    };
    if max_speed * e.dt / e.dx > FRAC_1_SQRT_2 {
        return Err("timestep violates the two-dimensional CFL bound".into());
    }
    for &[x, z] in &e.shots {
        if !x.is_finite()
            || !z.is_finite()
            || x < 0.0
            || z < 0.0
            || x > (e.nx - 1) as f64
            || z > (e.nz - 1) as f64
        {
            return Err("shot lies outside the grid".into());
        }
    }
    for &[x, z] in &e.receivers {
        if x >= e.nx || z >= e.nz {
            return Err("receiver lies outside the grid".into());
        }
    }

    let weights = if mode == "adjoint" {
        read_born_data(
            data_path.as_deref().unwrap(),
            &[e.shots.len(), e.steps, e.receivers.len()],
        )?
    } else {
        Vec::new()
    };
    fs::create_dir_all(&out)?;
    let mut traces = if mode == "adjoint" {
        None
    } else {
        let name = if mode == "born" {
            "born_data.npy"
        } else {
            "traces.npy"
        };
        let mut file = BufWriter::new(File::create(out.join(name))?);
        npy_header(
            &mut file,
            "<f8",
            &[e.shots.len(), e.steps, e.receivers.len()],
        )?;
        Some(file)
    };
    let mut wavefield = if let Some(every) = recording_every {
        let mut file = BufWriter::new(File::create(out.join("wavefield.npy"))?);
        npy_header(&mut file, "<f4", &[e.steps / every, e.nz, e.nx])?;
        Some(file)
    } else {
        None
    };

    let cells = e.nx * e.nz;
    let dt_sq = e.dt * e.dt;
    let inv_dx_sq = 1.0 / (e.dx * e.dx);
    let mut damping_dt = vec![0.0; cells];
    for z in 0..e.nz {
        for x in 0..e.nx {
            let distance = x.min(z).min(e.nx - 1 - x).min(e.nz - 1 - z) as f64;
            let taper = (1.0 - distance / e.sponge_width as f64).max(0.0);
            damping_dt[z * e.nx + x] = e.dt * e.sponge_strength * taper * taper;
        }
    }
    let mut previous = vec![0.0; cells];
    let mut current = vec![0.0; cells];
    let mut next = vec![0.0; cells];
    let mut dprevious = if mode == "born" {
        vec![0.0; cells]
    } else {
        Vec::new()
    };
    let mut dcurrent = dprevious.clone();
    let mut dnext = dprevious.clone();
    let mut image = if mode == "adjoint" {
        vec![0.0; cells]
    } else {
        Vec::new()
    };
    let mut footprint = vec![0.0; cells];
    println!("shot\tmode\tdata L2 norm");
    for (shot_index, &[shot_x, shot_z]) in e.shots.iter().enumerate() {
        previous.fill(0.0);
        current.fill(0.0);
        next.fill(0.0);
        dprevious.fill(0.0);
        dcurrent.fill(0.0);
        dnext.fill(0.0);
        let mut history = if mode == "adjoint" {
            vec![(previous.clone(), current.clone())]
        } else {
            Vec::new()
        };
        for z in 1..e.nz - 1 {
            for x in 1..e.nx - 1 {
                let radius_sq = (x as f64 - shot_x).powi(2) + (z as f64 - shot_z).powi(2);
                footprint[z * e.nx + x] = (-0.5 * radius_sq).exp();
            }
        }
        let mut norm_sq = 0.0;
        for step in 0..e.steps {
            let pulse = ricker(e, step);
            let tangent = if mode == "born" {
                Some((
                    &dprevious[..],
                    &dcurrent[..],
                    &perturbation[..],
                    &mut dnext[..],
                ))
            } else {
                None
            };
            advance(
                &previous,
                &current,
                &speed,
                &damping_dt,
                &footprint,
                &mut next,
                tangent,
                e.nx,
                e.nz,
                inv_dx_sq,
                dt_sq,
                pulse,
            );
            if mode != "adjoint" {
                let data = if mode == "born" { &dnext } else { &next };
                for &[x, z] in &e.receivers {
                    let sample = data[z * e.nx + x];
                    traces.as_mut().unwrap().write_all(&sample.to_le_bytes())?;
                    norm_sq += sample * sample;
                }
                if shot_index == 0 && recording_every.is_some_and(|every| (step + 1) % every == 0) {
                    let file = wavefield.as_mut().unwrap();
                    for &sample in data {
                        file.write_all(&(sample as f32).to_le_bytes())?;
                    }
                }
            }
            std::mem::swap(&mut previous, &mut current);
            std::mem::swap(&mut current, &mut next);
            if mode == "born" {
                std::mem::swap(&mut dprevious, &mut dcurrent);
                std::mem::swap(&mut dcurrent, &mut dnext);
            }
            if mode == "adjoint" {
                history.push((previous.clone(), current.clone()));
            }
        }
        if mode == "adjoint" {
            let shot_data = &weights[shot_index * e.steps * e.receivers.len()
                ..(shot_index + 1) * e.steps * e.receivers.len()];
            norm_sq = shot_data.iter().map(|value| value * value).sum();
            let mut adj_previous = vec![0.0; cells];
            let mut adj_current = vec![0.0; cells];
            let mut grad_previous = vec![0.0; cells];
            let mut grad_current = vec![0.0; cells];
            let mut grad_speed = vec![0.0; cells];
            for step in (0..e.steps).rev() {
                for (receiver, &[x, z]) in e.receivers.iter().enumerate() {
                    adj_current[z * e.nx + x] += shot_data[step * e.receivers.len() + receiver];
                }
                if shot_index == 0 && recording_every.is_some_and(|every| (step + 1) % every == 0) {
                    let file = wavefield.as_mut().unwrap();
                    for &sample in &adj_current {
                        file.write_all(&(sample as f32).to_le_bytes())?;
                    }
                }
                grad_previous.fill(0.0);
                grad_current.fill(0.0);
                grad_speed.fill(0.0);
                next.copy_from_slice(&history[step + 1].1);
                reverse_step(
                    &history[step].0,
                    &mut grad_previous,
                    &history[step].1,
                    &mut grad_current,
                    &speed,
                    &mut grad_speed,
                    &damping_dt,
                    &footprint,
                    &mut next,
                    &mut adj_current,
                    e.nx,
                    e.nz,
                    inv_dx_sq,
                    dt_sq,
                    ricker(e, step),
                );
                for i in 0..cells {
                    image[i] += grad_speed[i];
                    grad_current[i] += adj_previous[i];
                }
                std::mem::swap(&mut adj_previous, &mut grad_previous);
                std::mem::swap(&mut adj_current, &mut grad_current);
            }
        }
        println!("{shot_index}\t{mode}\t{:.12e}", norm_sq.sqrt());
    }
    if let Some(file) = traces.as_mut() {
        file.flush()?;
    }
    if let Some(file) = wavefield.as_mut() {
        file.flush()?;
    }
    if mode == "adjoint" {
        let mut file = BufWriter::new(File::create(out.join("image.npy"))?);
        npy_header(&mut file, "<f8", &[e.nz, e.nx])?;
        for &value in &image {
            file.write_all(&value.to_le_bytes())?;
        }
        file.flush()?;
    }

    let mut experiment_info = raw
        .as_object()
        .ok_or("experiment must be a JSON object")?
        .clone();
    experiment_info.remove("background");
    experiment_info.remove("perturbation");
    let mut run = json!({
        "experiment_file": experiment_path.to_str().ok_or("experiment path must be UTF-8")?,
        "experiment": experiment_info,
    });
    if let Some(every) = recording_every {
        let mut steps: Vec<usize> = (every..=e.steps).step_by(every).collect();
        if mode == "adjoint" {
            steps.reverse();
        }
        let times: Vec<f64> = steps.iter().map(|&step| step as f64 * e.dt).collect();
        run["recording"] = json!({"every": every, "steps": steps, "times": times});
    }
    write_json(&out.join("run.json"), &run)?;
    let mut result = json!({
        "mode": mode, "nx": e.nx, "nz": e.nz, "dx": e.dx, "dt": e.dt,
        "steps": e.steps, "shots": e.shots, "receivers": e.receivers,
    });
    if mode == "adjoint" {
        result["statistics"] = json!({
            "storage": "full",
            "checkpoints": null,
            "reverse_calls": e.shots.len() * e.steps,
            "scheduler_forward_calls": e.shots.len() * e.steps,
            "peak_saved_states": e.steps + 1,
            "peak_saved_bytes": (e.steps + 1) * 2 * cells * 8,
            "per_shot": vec![json!({
                "reverse_calls": e.steps,
                "scheduler_forward_calls": e.steps,
                "peak_saved_states": e.steps + 1,
            }); e.shots.len()],
        });
    }
    write_json(&out.join("result.json"), &result)?;
    Ok(())
}
