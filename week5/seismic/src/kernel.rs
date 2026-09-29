#![no_std]
#![feature(autodiff)]

use core::autodiff::{autodiff_forward, autodiff_reverse};

#[autodiff_forward(
    step_jvp, Dual, Dual, Dual, Const, Const, Dual, Const, Const, Const, Const, Const
)]
#[autodiff_reverse(
    step_vjp, Duplicated, Duplicated, Duplicated, Const, Const, Duplicated, Const, Const, Const,
    Const, Const
)]
fn step(
    previous: &[f64],
    current: &[f64],
    speed: &[f64],
    damping_dt: &[f64],
    footprint: &[f64],
    next: &mut [f64],
    nx: usize,
    nz: usize,
    inv_dx_sq: f64,
    dt_sq: f64,
    pulse: f64,
) {
    for z in 1..nz - 1 {
        for x in 1..nx - 1 {
            let i = z * nx + x;
            let laplacian = (current[i - 1] + current[i + 1] + current[i - nx] + current[i + nx]
                - 4.0 * current[i])
                * inv_dx_sq;
            let damping = damping_dt[i];
            next[i] = (2.0 * current[i] - (1.0 - damping) * previous[i]
                + dt_sq * (speed[i] * speed[i] * laplacian + pulse * footprint[i]))
                / (1.0 + damping);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn seismic_step_vjp(
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
) {
    let cells = nx * nz;
    unsafe {
        step_vjp(
            core::slice::from_raw_parts(previous, cells),
            core::slice::from_raw_parts_mut(dprevious, cells),
            core::slice::from_raw_parts(current, cells),
            core::slice::from_raw_parts_mut(dcurrent, cells),
            core::slice::from_raw_parts(speed, cells),
            core::slice::from_raw_parts_mut(dspeed, cells),
            core::slice::from_raw_parts(damping_dt, cells),
            core::slice::from_raw_parts(footprint, cells),
            core::slice::from_raw_parts_mut(next, cells),
            core::slice::from_raw_parts_mut(dnext, cells),
            nx,
            nz,
            inv_dx_sq,
            dt_sq,
            pulse,
        );
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn seismic_step(
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
) {
    let cells = nx * nz;
    unsafe {
        step(
            core::slice::from_raw_parts(previous, cells),
            core::slice::from_raw_parts(current, cells),
            core::slice::from_raw_parts(speed, cells),
            core::slice::from_raw_parts(damping_dt, cells),
            core::slice::from_raw_parts(footprint, cells),
            core::slice::from_raw_parts_mut(next, cells),
            nx,
            nz,
            inv_dx_sq,
            dt_sq,
            pulse,
        );
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn seismic_step_jvp(
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
) {
    let cells = nx * nz;
    unsafe {
        step_jvp(
            core::slice::from_raw_parts(previous, cells),
            core::slice::from_raw_parts(dprevious, cells),
            core::slice::from_raw_parts(current, cells),
            core::slice::from_raw_parts(dcurrent, cells),
            core::slice::from_raw_parts(speed, cells),
            core::slice::from_raw_parts(dspeed, cells),
            core::slice::from_raw_parts(damping_dt, cells),
            core::slice::from_raw_parts(footprint, cells),
            core::slice::from_raw_parts_mut(next, cells),
            core::slice::from_raw_parts_mut(dnext, cells),
            nx,
            nz,
            inv_dx_sq,
            dt_sq,
            pulse,
        );
    }
}

unsafe extern "C" {
    fn abort() -> !;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { abort() }
}
