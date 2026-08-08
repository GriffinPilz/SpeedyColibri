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

/// int2 (`fmt 3`): 4 values/byte, `value = field - 2`, one f32 scale per row.
///
/// **This format had no row in either table until Maple shipped on it** — the same
/// closed-set gap that once left MXFP4, the format two models depend on, impossible to
/// microbenchmark. `qtensor_from_f32` at `bits = 2` routes to `pack_int2`, which is the
/// encoder the ternary converter uses, so the benchmark cannot encode in a way the kernels
/// do not decode.
fn int2_tensor(o: usize, i: usize) -> QTensor {
    let w: Vec<f32> = (0..o * i).map(|k| ((k as f32) * 0.017).sin()).collect();
    let mut t = crate::quantize::qtensor_from_f32(&w, o, i, 2);
    t.gpu_eligible = true;
    t
}

/// Quantize `w` [o, i] to MXFP4 (fmt 6): E2M1 nibbles, two per byte with the EVEN column
/// in the low nibble, plus one E8M0 (bare power-of-two) scale per **32** inputs.
///
/// **Benchmark-only, and deliberately not in `convert.rs`:** nothing in the pipeline ever
/// quantizes *to* MXFP4. V4's and K3's routed experts arrive QAT-trained in MXFP4 and
/// `mxfp4_passthrough_out` copies them bit-exact, because a dequant→requantize round trip
/// measured 6.40% rel-RMS of pure loss. This exists only to build a weight of the right
/// shape and format so the kernel can be timed. `g = 1.0` matches what that passthrough
/// writes to the `.mx` sidecar — MXFP4 has no global scale, the block scale carries it.
///
/// The nibble encoding is `convert::e2m1_code`, the same function the NVFP4 quantizer
/// uses, so the benchmark cannot encode in a way the kernels do not decode.
fn quantize_mxfp4(w: &[f32], o: usize, i: usize) -> (Vec<u8>, Vec<u8>) {
    const GS: usize = 32;
    let nb = i.div_ceil(GS);
    let rb = i.div_ceil(2);
    let mut q4 = vec![0u8; o * rb];
    let mut bs = vec![0u8; o * nb];
    for r in 0..o {
        for b in 0..nb {
            let c0 = b * GS;
            let c1 = ((b + 1) * GS).min(i);
            let bmax = w[r * i + c0..r * i + c1].iter().fold(0f32, |m, &v| m.max(v.abs()));
            // E8M0 encodes a bare power of two 2^(e-127) — there is no mantissa to absorb
            // a remainder, so pick the smallest e whose scale brings the block max onto
            // E2M1's top level (6.0) or below. `ceil` (not `round`) is what guarantees
            // that: rounding down would clip the block's largest weight.
            let e = if bmax > 0.0 {
                (bmax / 6.0).log2().ceil() as i32
            } else {
                0
            };
            let e = e.clamp(-127, 127);
            bs[r * nb + b] = (127 + e) as u8;
            let sc = (e as f32).exp2();
            for c in c0..c1 {
                let code = crate::convert::e2m1_code(w[r * i + c] / sc);
                let idx = r * rb + (c >> 1);
                if c & 1 == 1 {
                    q4[idx] |= code << 4;
                } else {
                    q4[idx] |= code;
                }
            }
        }
    }
    (q4, bs)
}

fn mxfp4_tensor(o: usize, i: usize) -> QTensor {
    let w: Vec<f32> = (0..o * i).map(|k| ((k as f32) * 0.017).sin()).collect();
    let (q4, bs) = quantize_mxfp4(&w, o, i);
    QTensor {
        fmt_code: 6,
        o: o as i32,
        i: i as i32,
        q4: Bytes::Owned(q4),
        bs: Bytes::Owned(bs),
        g: 1.0,
        gpu_eligible: true,
        ..Default::default()
    }
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
        // Maple's decode-path resident shapes (ternary/int2). `lm_head` is deliberately
        // absent: at [151936, 2048] building it in three formats needs several GB of
        // scratch, and it is fmt 0 (f32) in the container anyway — no int2/nvfp4 arm to
        // compare it against.
        ("maple-qkv", 3072, 2048),
        ("maple-o", 2048, 2048),
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
            [
                (name, o, i, "nvfp4", nvfp4_tensor(o, i)),
                (name, o, i, "int8", int8_tensor(o, i)),
                (name, o, i, "int2", int2_tensor(o, i)),
            ]
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

/// bf16 (`fmt 2`): raw 2-byte values in `q4`, no per-row scale — the IO tier.
fn bf16_tensor(o: usize, i: usize) -> QTensor {
    let mut bytes = Vec::with_capacity(o * i * 2);
    for k in 0..o * i {
        let v = ((k as f32) * 0.017).sin();
        bytes.extend_from_slice(&colibri_core::f32_to_bf16(v).to_le_bytes());
    }
    QTensor {
        fmt_code: 2,
        o: o as i32,
        i: i as i32,
        q4: bytes.into(),
        gpu_eligible: true,
        ..Default::default()
    }
}

/// f32 (`fmt 0`): the IO tier bf16 replaced. Here so the two can be compared directly.
fn f32_tensor(o: usize, i: usize) -> QTensor {
    QTensor {
        fmt_code: 0,
        o: o as i32,
        i: i as i32,
        qf: (0..o * i).map(|k| ((k as f32) * 0.017).sin()).collect(),
        gpu_eligible: true,
        ..Default::default()
    }
}

/// The `lm_head` shape, in the three tiers it can ship in, measured in ISOLATION.
///
/// This exists because the phase timer could not settle the question. `lm_head` is the
/// largest single per-token read in any fits-RAM model here, and `logits_us` put it at
/// ~145 GB/s against a measured 257 GB/s streaming ceiling — but four different kernel
/// shapes (block reduction, warp-per-row narrow, warp-per-row wide, and a rows-per-block
/// sweep) all came back within noise of each other, which is what you see either when the
/// kernel is already at a hardware limit or when the number is not really the kernel's.
/// A phase timer cannot tell those apart; a direct per-call measurement can.
///
/// `s = 1` on purpose: this is the decode regime, where the row count is what selects the
/// GEMV in the first place.
pub fn io_report(reps: usize) {
    let (o, i) = (151936usize, 2048usize);
    println!();
    println!("lm_head [{o}, {i}] per-call cost   S=1  reps={reps}");
    match ceiling_gbs() {
        Some(c) => println!("  streaming-read ceiling = {c:.0} GB/s"),
        None => println!("  streaming-read ceiling = unavailable (no CUDA)"),
    }
    println!(
        "  {:<7} {:>10} {:>9} {:>9} {:>10}  {}",
        "tier", "us/call", "MB", "GB/s", "% ceiling", "on-gpu"
    );
    let ceil = ceiling_gbs().unwrap_or(f64::NAN);
    // Built one at a time and dropped: three tiers of a 311M-parameter weight is ~2 GB of
    // host memory held at once otherwise. `time_one` warms each before timing, so the
    // device registration is not folded in.
    for tier in ["bf16", "int8", "f32"] {
        let t = match tier {
            "bf16" => bf16_tensor(o, i),
            "int8" => int8_tensor(o, i),
            _ => f32_tensor(o, i),
        };
        let (us, frac) = time_one(&t, 1, reps);
        let mb = t.bytes() as f64 / 1e6;
        let gbs = mb / us * 1e3;
        println!(
            "  {:<7} {:>10.1} {:>9.1} {:>9.0} {:>9.0}%  {}",
            tier,
            us,
            mb,
            gbs,
            100.0 * gbs / ceil,
            if frac > 0.99 { "yes".to_string() } else { format!("NO ({:.0}%)", frac * 100.0) }
        );
    }
}

/// One measured expert FFN — the fused gate+up+down triple, not a single matmul.
pub struct ExpertRow {
    pub name: &'static str,
    /// Expert input width (`gate.i`), i.e. the model hidden or MoE latent.
    pub d: usize,
    /// Expert intermediate width (`gate.o`).
    pub i: usize,
    pub fmt: &'static str,
    pub us_per_call: f64,
    /// gate+up+down weight bytes touched per call.
    pub bytes: i64,
    pub gpu_frac: f64,
}

/// Time `reps` fused expert FFNs. Returns (µs/call, fraction that reached the GPU).
fn time_expert(
    gate: &QTensor,
    up: &QTensor,
    down: &QTensor,
    s: usize,
    reps: usize,
) -> (f64, f64) {
    let d = gate.i as usize;
    let x: Vec<f32> = (0..s * d).map(|k| ((k as f32) * 0.011).cos()).collect();
    let mut y = vec![0f32; s * d];
    let call = |y: &mut [f32]| {
        if gate.fmt_code == 6 {
            crate::gpu::try_expert_ffn_mxfp4(gate, up, down, &x, s, y)
        } else {
            crate::gpu::try_expert_ffn(gate, up, down, &x, s, y)
        }
    };
    call(&mut y); // warm the wrap, as in `time_one`
    let c0 = crate::gpu::ffn_count();
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        call(&mut y);
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
    std::hint::black_box(&y);
    (us, (crate::gpu::ffn_count() - c0) as f64 / reps as f64)
}

/// Sweep the routed-expert FFN at real expert shapes, for **both** 4-bit formats.
///
/// This exists because MXFP4 was unmeasurable. `sweep_shapes` above builds nvfp4 and int8
/// only, and MXFP4 cannot join it: `matches!(kind, Kind::X)` in `convert.rs` means fmt 6 is
/// only ever a routed *expert*, so there is no dense `coli_cuda_matmul_mxfp4` for
/// `matmul_qt` to call — a naive fmt-6 row would have timed the single-threaded CPU
/// fallback and been marked `NO`. Adding that dense kernel purely to feed a benchmark would
/// ship a production code path nothing calls, and would measure a shape V4 and K3 never use.
///
/// So this measures the entry point that *does* exist — and measures NVFP4 through its own
/// expert entry point on the same shapes, because a table with one format in it answers
/// nothing. That comparison is the whole value of the matmul table above: it is what
/// established that 4-bit is dequant-bound rather than bandwidth-bound (~190 GB/s vs int8's
/// 400-556 on the same shape, slower in absolute time while reading half the bytes).
///
/// Numbers here are NOT comparable to `sweep`'s: one call is three GEMMs plus an activation
/// plus one host round trip, so it is reported in its own table.
pub fn expert_sweep(s: usize, reps: usize) -> Vec<ExpertRow> {
    // (name, D, I) — gate/up are [I, D], down is [D, I]. Both are the real routed-expert
    // shapes, so the floor is compared against calls that actually happen.
    const SHAPES: &[(&str, usize, usize)] = &[
        ("floor", 128, 64),
        ("v4-expert", 4096, 2048),  // DeepSeek-V4: hidden 4096, moe_intermediate 2048
        ("k3-expert", 3584, 3072),  // Kimi-K3: routed_expert_hidden 3584, moe_inter 3072
        ("maple-expert", 2048, 512), // Maple: hidden 2048, moe_intermediate 512
    ];
    // The fused kernels apply whatever `gpu::set_activation` last set. Pin it so a run is
    // reproducible and both formats do the same arithmetic; plain SiLU, no clamp.
    crate::gpu::set_activation(false, 0.0, 0.0);
    let built: Vec<(&'static str, usize, usize, &'static str, [QTensor; 3])> = SHAPES
        .iter()
        .flat_map(|&(name, d, i)| {
            [
                (name, d, i, "nvfp4", [nvfp4_tensor(i, d), nvfp4_tensor(i, d), nvfp4_tensor(d, i)]),
                (name, d, i, "mxfp4", [mxfp4_tensor(i, d), mxfp4_tensor(i, d), mxfp4_tensor(d, i)]),
                (name, d, i, "int2", [int2_tensor(i, d), int2_tensor(i, d), int2_tensor(d, i)]),
            ]
        })
        .collect();
    built
        .iter()
        .map(|(name, d, i, fmt, t)| {
            let (us_per_call, gpu_frac) = time_expert(&t[0], &t[1], &t[2], s, reps);
            ExpertRow {
                name,
                d: *d,
                i: *i,
                fmt,
                us_per_call,
                bytes: t[0].bytes() + t[1].bytes() + t[2].bytes(),
                gpu_frac,
            }
        })
        .collect()
}

/// Print the expert-FFN sweep. Same over-floor convention as [`report`].
pub fn report_experts(s: usize, reps: usize) {
    let rows = expert_sweep(s, reps);
    let floor = rows
        .iter()
        .filter(|r| r.name == "floor")
        .map(|r| r.us_per_call)
        .fold(f64::INFINITY, f64::min);
    println!();
    println!("expert FFN per-call cost (fused gate+up+down)   S={s}  reps={reps}");
    println!("  floor (128x64, negligible arithmetic) = {floor:.1} us/call");
    println!("  NOT comparable to the matmul table above: one call is 3 GEMMs + activation.");
    println!();
    println!(
        "  {:<11} {:>6} {:>6} {:>7} {:>10} {:>11} {:>9} {:>8}  {}",
        "shape", "D", "I", "fmt", "us/call", "over-floor", "MB", "GB/s", "on-gpu"
    );
    for r in &rows {
        let mb = r.bytes as f64 / 1e6;
        let over = (r.us_per_call - floor).max(0.0);
        let gbs = if over > 0.0 { mb / over * 1e3 } else { f64::NAN };
        println!(
            "  {:<11} {:>6} {:>6} {:>7} {:>10.1} {:>11.1} {:>9.1} {:>8.0}  {}",
            r.name,
            r.d,
            r.i,
            r.fmt,
            r.us_per_call,
            over,
            mb,
            gbs,
            if r.gpu_frac > 0.99 {
                "yes".to_string()
            } else {
                format!("NO ({:.0}%)", r.gpu_frac * 100.0)
            }
        );
    }
    if rows.iter().any(|r| r.gpu_frac <= 0.99) {
        println!();
        println!("  NOTE: a row marked NO never reached a CUDA kernel. The MXFP4 rows need");
        println!("  zero-copy (`COLI_NO_ZEROCOPY=1` disables it); both need a CUDA device.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The benchmark's MXFP4 weights must decode the way a real container's do, or the
    /// fmt-6 rows time a kernel on data no model could contain. Checks the round trip
    /// through `matmul_qt`'s fmt-6 arm — the same decode the loader and CUDA kernels
    /// agree with, and itself pinned by `linear::tests::matmul_qt_reconstructs_mxfp4`.
    ///
    /// An identity input recovers the whole matrix in one call: with `nr = i` and
    /// `x = I`, `y[k*o + r] == W_q[r, k]`.
    ///
    /// **The weights here are adversarial on purpose, and the first version of this test
    /// was not.** With the benchmark's own smooth `sin(k * 0.017)`, adjacent columns are
    /// nearly equal, so swapping which nibble of a byte holds which column changes almost
    /// nothing — that mutation PASSED. Damping the odd columns to ~1/8 makes an even/odd
    /// swap a large error, and the mutation then fails as it should.
    #[test]
    fn quantize_mxfp4_round_trips_through_the_production_decode() {
        let (o, i) = (32usize, 64usize); // 2 blocks of 32 per row
        let w: Vec<f32> = (0..o * i)
            .map(|k| {
                let v = ((k as f32) * 0.017).sin();
                if k % 2 == 0 {
                    v
                } else {
                    v * 0.12
                }
            })
            .collect();
        let (q4, bs) = quantize_mxfp4(&w, o, i);
        let t = QTensor {
            fmt_code: 6,
            o: o as i32,
            i: i as i32,
            q4: Bytes::Owned(q4),
            bs: Bytes::Owned(bs),
            g: 1.0,
            ..Default::default() // gpu_eligible false => the CPU reference decode
        };
        let mut x = vec![0f32; i * i];
        for k in 0..i {
            x[k * i + k] = 1.0;
        }
        let mut y = vec![0f32; i * o];
        crate::linear::matmul_qt(&mut y, &x, &t, i);

        let (mut num, mut den) = (0f64, 0f64);
        for r in 0..o {
            for k in 0..i {
                let d = (y[k * o + r] - w[r * i + k]) as f64;
                num += d * d;
                den += (w[r * i + k] as f64).powi(2);
            }
        }
        let rel = (num / den).sqrt();
        // 4-bit with a per-32 block scale lands around 0.1. A swapped nibble order or a
        // block stride of 16 (NVFP4's) decodes essentially uncorrelated values and lands
        // near 1.0, so this bound separates "quantized" from "scrambled".
        assert!(
            rel < 0.25,
            "MXFP4 round-trip rel-RMS {rel:.3}: too high for 4-bit — suspect nibble order \
             or a block stride that is not 32"
        );
        // Guard against the test passing by comparing something with itself: 4 bits
        // cannot be this accurate.
        assert!(
            rel > 1e-4,
            "rel-RMS {rel:.6} is impossibly good for 4 bits — the reference is not independent"
        );
    }
}

/// Print a sweep with the floor called out explicitly. The interesting column is
/// `over-floor`: time that is not the fixed round trip.
/// Measured streaming-read ceiling of device memory, GB/s.
///
/// 1 GiB, which is far past any cache on this part, so it is DRAM and not L2. Reported
/// beside every achieved GB/s because "we get 141 GB/s" means nothing on its own: against
/// the 273 GB/s on the spec sheet it reads as half-wasted, and against what a do-nothing
/// read kernel actually reaches it may be essentially done.
#[cfg(feature = "cuda")]
pub fn ceiling_gbs() -> Option<f64> {
    colibri_backend::cuda::bandwidth_gbs(1 << 30, 5)
}
#[cfg(not(feature = "cuda"))]
pub fn ceiling_gbs() -> Option<f64> {
    None
}

pub fn report(s: usize, reps: usize) {
    let rows = sweep(s, reps);
    let floor = rows.iter().filter(|r| r.name == "floor").map(|r| r.us_per_call).fold(f64::INFINITY, f64::min);
    println!("gpu matmul per-call cost   S={s}  reps={reps}");
    println!("  floor (64x128, negligible arithmetic) = {floor:.1} us/call");
    match ceiling_gbs() {
        Some(c) => println!(
            "  streaming-read ceiling (1 GiB, no arithmetic)  = {c:.0} GB/s  \
             <- compare the GB/s column to THIS, not to the spec sheet"
        ),
        None => println!("  streaming-read ceiling = unavailable (no CUDA)"),
    }
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
