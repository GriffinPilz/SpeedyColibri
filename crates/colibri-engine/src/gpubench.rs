//! Per-call GPU matmul cost, isolated from the model.
//!
//! Decode issues one `matmul_qt` per dense weight per layer at `S == 1`, and every
//! one of them is a blocking `cudaMemcpy` HtoD, a launch, a `cudaMemcpyAsync` DtoH
//! and a `cudaStreamSynchronize` (`coli_cuda_matmul*` in `backend_cuda.cu`). That
//! fixed cost is invisible in a phase profile — it is spread across `mamba`, `moe`
//! and `attn` alike — so it has to be measured on its own.
//!
//! The method is a **shape sweep, not a single number**: a deliberately tiny weight
//! does no meaningful work, so whatever time it takes IS the per-call floor. Every
//! larger shape is then reported against that floor, which is what says whether a
//! given matmul is doing arithmetic or waiting on the round trip. A single shape
//! cannot distinguish the two.
//!
//! Shapes are Nemotron-3-Super's real decode shapes so the floor is being compared
//! against calls that actually happen, not against a synthetic ideal.

use colibri_core::quant::{Bytes, QTensor};

/// One measured shape.
pub struct Row {
    pub name: &'static str,
    pub o: usize,
    pub i: usize,
    pub fmt: &'static str,
    pub us_per_call: f64,
    /// Weight bytes touched per call — the read a perfect kernel would still pay.
    pub bytes: i64,
    /// Fraction of calls that actually reached the GPU. A CUDA entry point that
    /// returns 0 falls back to the single-threaded CPU matmul and reports nothing,
    /// so a slow row is ambiguous — "bad kernel" and "no kernel" look identical
    /// from the outside. Counting the dispatches disambiguates them.
    pub gpu_frac: f64,
}

fn nvfp4_tensor(o: usize, i: usize) -> QTensor {
    let w: Vec<f32> = (0..o * i).map(|k| ((k as f32) * 0.017).sin()).collect();
    let (q4, bs, g) = crate::convert::quantize_nvfp4(&w, o, i);
    QTensor {
        fmt_code: 5,
        o: o as i32,
        i: i as i32,
        q4: Bytes::Owned(q4),
        bs: Bytes::Owned(bs),
        g,
        gpu_eligible: true,
        ..Default::default()
    }
}

fn int8_tensor(o: usize, i: usize) -> QTensor {
    let w: Vec<f32> = (0..o * i).map(|k| ((k as f32) * 0.017).sin()).collect();
    let mut t = crate::quantize::qtensor_from_f32(&w, o, i, 8);
    t.gpu_eligible = true;
    t
}

/// Time `reps` calls of `matmul_qt` at `S = s` for one weight. Returns
/// (µs/call, fraction of calls that reached the GPU).
fn time_one(t: &QTensor, s: usize, reps: usize) -> (f64, f64) {
    let (i, o) = (t.i as usize, t.o as usize);
    let x: Vec<f32> = (0..s * i).map(|k| ((k as f32) * 0.011).cos()).collect();
    let mut y = vec![0f32; s * o];
    // Warm the wrap: the first call registers the weight in the thread-local
    // resident map and allocates the device scratch. Timing it would fold a
    // one-off setup into a per-call figure.
    crate::linear::matmul_qt(&mut y, &x, t, s);
    let g0 = crate::gpu::matmul_count();
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        crate::linear::matmul_qt(&mut y, &x, t, s);
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
    // Consume `y` so the loop cannot be optimized away.
    std::hint::black_box(&y);
    (us, (crate::gpu::matmul_count() - g0) as f64 / reps as f64)
}

/// Sweep Nemotron's decode shapes plus a floor probe. `s` is the row count
/// (1 = decode).
pub fn sweep(s: usize, reps: usize) -> Vec<Row> {
    // (name, O, I). `floor` is small enough that its arithmetic is negligible on
    // any GPU, so its time is the per-call overhead and nothing else.
    const SHAPES: &[(&str, usize, usize)] = &[
        ("floor", 64, 128),
        ("moe-fc1", 1024, 4096),
        ("moe-fc2", 4096, 1024),
        ("mamba-in", 10240, 4096),
        ("mamba-out", 4096, 8192),
        ("attn-qkv", 5120, 4096),
        // Bracket around moe-fc2 (4096x1024): vary O at fixed I, then I at fixed O.
        // One anomalous point is a shape; a run of them is a rule.
        ("br-O2048", 2048, 1024),
        ("br-O8192", 8192, 1024),
        ("br-I512", 4096, 512),
        ("br-I2048", 4096, 2048),
    ];
    sweep_shapes(SHAPES, s, reps)
}

/// The sweep over an explicit shape list, so a suspicious row can be bracketed
/// without editing the default set.
///
/// Every tensor is built up front and held alive for the whole sweep. `gpu.rs`
/// keys its device-tensor registry on the **host weight address**, which is sound
/// for a model (weights outlive the process) but not for a loop that drops each
/// tensor before building the next: the allocator reuses the address and the next
/// shape gets the previous shape's device tensor. That mismatch makes the CUDA
/// call fail, which is silent — it falls back to the CPU matmul. Measured that
/// way, half this table was timing the fallback.
pub fn sweep_shapes(shapes: &[(&'static str, usize, usize)], s: usize, reps: usize) -> Vec<Row> {
    let built: Vec<(&'static str, usize, usize, &'static str, QTensor)> = shapes
        .iter()
        .flat_map(|&(name, o, i)| {
            [(name, o, i, "nvfp4", nvfp4_tensor(o, i)), (name, o, i, "int8", int8_tensor(o, i))]
        })
        .collect();
    built
        .iter()
        .map(|(name, o, i, fmt, t)| {
            let (us_per_call, gpu_frac) = time_one(t, s, reps);
            Row { name, o: *o, i: *i, fmt, us_per_call, bytes: t.bytes(), gpu_frac }
        })
        .collect()
}

/// Print a sweep with the floor called out explicitly. The interesting column is
/// `over-floor`: time that is not the fixed round trip.
pub fn report(s: usize, reps: usize) {
    let rows = sweep(s, reps);
    let floor = rows.iter().filter(|r| r.name == "floor").map(|r| r.us_per_call).fold(f64::INFINITY, f64::min);
    println!("gpu matmul per-call cost   S={s}  reps={reps}");
    println!("  floor (64x128, negligible arithmetic) = {floor:.1} us/call");
    println!();
    println!("  {:<10} {:>6} {:>6} {:>7} {:>10} {:>11} {:>9} {:>8}  {}", "shape", "O", "I", "fmt", "us/call", "over-floor", "MB", "GB/s", "on-gpu");
    for r in &rows {
        let mb = r.bytes as f64 / 1e6;
        let over = (r.us_per_call - floor).max(0.0);
        // Bandwidth is charged against time-over-floor: a kernel cannot be blamed
        // for the round trip, and crediting it with those microseconds would make a
        // latency-bound call look like a fast one.
        let gbs = if over > 0.0 { mb / over * 1e3 } else { f64::NAN };
        println!(
            "  {:<10} {:>6} {:>6} {:>7} {:>10.1} {:>11.1} {:>9.1} {:>8.0}  {}",
            r.name, r.o, r.i, r.fmt, r.us_per_call, over, mb, gbs,
            if r.gpu_frac > 0.99 { "yes".to_string() } else { format!("NO ({:.0}%)", r.gpu_frac * 100.0) }
        );
    }
    if rows.iter().any(|r| r.gpu_frac <= 0.99) {
        println!();
        println!("  NOTE: a row marked NO never reached a CUDA kernel — it ran the single-threaded");
        println!("  CPU matmul. Its time says nothing about GPU performance, only about the fallback.");
    }
}
