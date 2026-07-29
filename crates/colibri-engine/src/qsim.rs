//! Load-time **simulated requantization** of resident weights — `COLI_QSIM`.
//!
//! Answers "what would this model's quality be if tensor class C were stored at
//! precision P?" without converting a container or writing a kernel. Each selected
//! tensor is dequantized, round-tripped through the target scheme's numerics, and
//! re-quantized into its original format — so the *values* carry P's error while the
//! storage layout and every kernel stay exactly as they are.
//!
//! This exists because the alternative is hours of requantization per data point, which
//! makes a real sensitivity sweep unaffordable and quantization choices a matter of
//! taste. With it, `coli ppl <snap> <text>` prices any assignment in minutes.
//!
//! # Usage
//!
//! ```text
//! COLI_QSIM=mamba:nvfp4                 # Mamba projections at NVFP4
//! COLI_QSIM=mamba:6                     # ... at 6-bit per-row int
//! COLI_QSIM=mamba_in:nvfp4,shared:6     # per-class hybrid — the real question
//! COLI_QSIM=resident:nvfp4              # every resident 2-D weight
//! ```
//!
//! Classes: `mamba` (= `mamba_in` + `mamba_out`), `mamba_in`, `mamba_out`, `latent`
//! (fc1/fc2), `shared` (shared-expert + dense MLP), `attn` (q/k/v/o, MLA a/b, fused
//! qkv), `resident` / `all`. Schemes: `nvfp4`, or an integer bit width (`8`…`2`).
//!
//! # What it does and does not measure
//!
//! **Measures**: the quality cost of the target precision, which is the thing that
//! decides whether an assignment is shippable.
//!
//! **Does not measure**: speed. Simulation keeps the original storage, so bytes moved
//! per token are unchanged — a `COLI_QSIM` run tells you nothing about tok/s. Pair it
//! with the byte arithmetic (`bits_target / bits_stored × class bytes`) for that half.
//!
//! # RMS does not rank formats — read before trusting a comparison
//!
//! Every rule reports the **relative RMS perturbation it actually applied**, and that
//! number exists to be checked against `coli qerr` for the same scheme. An arm whose rms
//! is far below the reference is near-identity, and its perplexity means nothing; the
//! tool warns below 1%.
//!
//! But the headline lesson from the first real sweep is different, and sharper:
//! **equal RMS across different formats does NOT mean equal damage.** Measured on
//! Nemotron-3-Super (6000 tokens, deterministic, baseline byte-identical twice):
//!
//! | resident scheme | rms applied | perplexity |
//! |---|---|---|
//! | baseline int8   | —       | 6.781 |
//! | nvfp4           | 0.0943  | **6.697** (unharmed) |
//! | int6            | 0.0596  | 7.044 (+3.9%) |
//! | int5            | 0.1176  | 7.075 (+4.3%) |
//!
//! NVFP4 perturbs *more* than int6 by RMS and costs *nothing*, while the int family
//! degrades monotonically. The reason is the error's **structure**, not its size: per-row
//! int puts one scale on a whole row, so each weight's error is proportional to that
//! row's max — small weights take large *relative* damage. NVFP4's per-16 block scales
//! make each weight's error proportional to its own local magnitude. Relative error
//! preserves the structure of small weights; absolute error destroys it.
//!
//! So do not rank schemes by `qerr` RMS across format families — it is valid *within* a
//! family (int8 < int6 < int5) and misleading *across* one. Rank by perplexity here.
//!
//! Remaining caveat: this simulates P on top of the container's existing int8. The
//! measured rms (0.0943) landing on the qerr-vs-source figure (0.0935) says the
//! compounding is mild, as `sqrt(int8² + P²)` predicts — but a shippable decision still
//! wants a real requant from source and more than one corpus.

use crate::model::{Layer, Model};
use colibri_core::QTensor;

/// Target precision for a simulated class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimScheme {
    /// NVFP4: e2m1 data, one ue4m3 scale per 16 inputs, plus a per-tensor f32 scale.
    Nvfp4,
    /// Per-row linear int-N with one f32 scale per output row.
    Int(u32),
}

/// Which resident tensors a rule applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    MambaIn,
    MambaOut,
    Latent,
    Shared,
    Attn,
    All,
}

impl Class {
    /// Parse one class token. `mamba` expands to both projections, so it returns a slice.
    fn parse(s: &str) -> Option<&'static [Class]> {
        Some(match s {
            "mamba" => &[Class::MambaIn, Class::MambaOut],
            "mamba_in" => &[Class::MambaIn],
            "mamba_out" => &[Class::MambaOut],
            "latent" => &[Class::Latent],
            "shared" => &[Class::Shared],
            "attn" => &[Class::Attn],
            "resident" | "all" => &[Class::All],
            _ => return None,
        })
    }
}

/// One parsed `class:scheme` rule.
struct Rule {
    class: Class,
    scheme: SimScheme,
}

/// Parse `COLI_QSIM`'s value: comma-separated `class:scheme` pairs. Returns the rules
/// and a human-readable echo. Unknown tokens are reported rather than ignored — a typo
/// that silently simulates nothing would look exactly like "this precision is free".
fn parse_rules(spec: &str) -> Result<Vec<Rule>, String> {
    let mut rules = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (cls, sch) = part
            .split_once(':')
            .ok_or_else(|| format!("bad COLI_QSIM entry {part:?} (want class:scheme)"))?;
        let classes = Class::parse(cls.trim()).ok_or_else(|| {
            format!(
                "unknown COLI_QSIM class {:?} (mamba|mamba_in|mamba_out|latent|shared|attn|resident)",
                cls.trim()
            )
        })?;
        let scheme = match sch.trim() {
            "nvfp4" => SimScheme::Nvfp4,
            n => SimScheme::Int(
                n.parse::<u32>()
                    .map_err(|_| format!("bad COLI_QSIM scheme {n:?} (want `nvfp4` or a bit count)"))
                    .and_then(|b| {
                        if (2..=16).contains(&b) {
                            Ok(b)
                        } else {
                            Err(format!("COLI_QSIM bit width {b} out of range 2..=16"))
                        }
                    })?,
            ),
        };
        for &class in classes {
            rules.push(Rule { class, scheme });
        }
    }
    Ok(rules)
}

/// Round-trip one tensor's values through `scheme`, re-storing in its original format.
///
/// Returns `(weights touched, Σ squared delta, Σ squared original)` so the caller can
/// report the **relative RMS perturbation actually applied**.
///
/// That number is not decoration — it is the tool's own validity check. An arm that
/// perturbed nothing produces a perplexity indistinguishable from baseline, which reads
/// as "this precision is free": the most expensive way this tool could mislead. Compare
/// the reported rms against `coli qerr` for the same scheme against the ORIGINAL source;
/// far lower means the arm is measuring noise, and the caller warns below 1%.
///
/// (This check earned its keep immediately: it refuted a plausible hypothesis that
/// `nvfp4-on-int8` was near-identity because the two grids interact. The measured rms
/// came out at 0.0943 against a 0.0935 reference — the perturbation was entirely real,
/// and the surprising perplexity result was a genuine format effect. See the module
/// header.)
fn simulate(t: &mut QTensor, scheme: SimScheme) -> (usize, f64, f64) {
    let (o, i) = (t.o as usize, t.i as usize);
    if o == 0 || i == 0 {
        return (0, 0.0, 0.0);
    }
    // Decide the re-store width FIRST, and bail on anything we cannot faithfully read.
    // `dequantize_qtensor` returns ZEROS for formats it doesn't decode (int2, nvfp4), so
    // dequantizing before this check would silently replace such a tensor with zeros —
    // a model-destroying no-op that reports success. Resident weights are f32 or int8;
    // anything else belongs to the streamed-expert path, which never comes through here.
    let bits = match t.fmt_code {
        0 => 32,
        1 => 8,
        _ => return (0, 0.0, 0.0),
    };
    let w = crate::convert::dequantize_qtensor_pub(t);
    let approx = match scheme {
        SimScheme::Nvfp4 => crate::convert::quantize_nvfp4_sim_pub(&w, o, i),
        SimScheme::Int(bits) => {
            crate::convert::dequantize_qtensor_pub(&crate::quantize::qtensor_from_f32(&w, o, i, bits))
        }
    };
    let mut re = crate::quantize::qtensor_from_f32(&approx, o, i, bits);
    re.gpu_eligible = t.gpu_eligible;
    // Perturbation actually delivered, measured against the values we started from
    // (post-restore, so it includes the re-quantization step the model will really see).
    let after = crate::convert::dequantize_qtensor_pub(&re);
    let (mut se, mut sr) = (0f64, 0f64);
    for (a, b) in after.iter().zip(&w) {
        se += ((a - b) as f64) * ((a - b) as f64);
        sr += (*b as f64) * (*b as f64);
    }
    *t = re;
    (o * i, se, sr)
}

/// Apply `scheme` to every tensor of `class` in one layer; returns weights touched.
fn apply_layer(l: &mut Layer, class: Class, scheme: SimScheme) -> (usize, f64, f64) {
    let (mut n, mut se, mut sr) = (0usize, 0f64, 0f64);
    let mut acc = |r: (usize, f64, f64)| {
        n += r.0;
        se += r.1;
        sr += r.2;
    };
    let mut opt = |t: &mut Option<QTensor>| {
        if let Some(t) = t.as_mut() {
            acc(simulate(t, scheme));
        }
    };
    match class {
        Class::MambaIn => opt(&mut l.mamba_in_proj),
        Class::MambaOut => opt(&mut l.mamba_out_proj),
        Class::Latent => {
            opt(&mut l.fc1_latent);
            opt(&mut l.fc2_latent);
        }
        Class::Shared => {
            for t in [&mut l.sh_gate, &mut l.sh_up, &mut l.sh_down, &mut l.gate_proj, &mut l.up_proj, &mut l.down_proj] {
                acc(simulate(t, scheme));
            }
        }
        Class::Attn => {
            opt(&mut l.q_proj);
            opt(&mut l.k_proj);
            opt(&mut l.v_proj);
            opt(&mut l.qkv_proj);
            for t in [&mut l.q_a, &mut l.q_b, &mut l.kv_a, &mut l.kv_b, &mut l.o] {
                acc(simulate(t, scheme));
            }
        }
        Class::All => {
            for c in [Class::MambaIn, Class::MambaOut, Class::Latent, Class::Shared, Class::Attn] {
                acc(apply_layer(l, c, scheme));
            }
        }
    }
    (n, se, sr)
}

/// Apply `COLI_QSIM` to a freshly loaded model. No-op when the env is unset.
///
/// Logs what it actually touched, including a loud warning when a rule matched zero
/// weights: a silent no-op would be indistinguishable from "that precision costs
/// nothing", which is the single most dangerous way this tool could mislead.
pub fn apply_qsim(model: &mut Model) {
    let Ok(spec) = std::env::var("COLI_QSIM") else { return };
    if spec.trim().is_empty() {
        return;
    }
    let rules = match parse_rules(&spec) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[qsim] {e} — ignoring COLI_QSIM, model served UNMODIFIED");
            return;
        }
    };
    let t0 = std::time::Instant::now();
    for rule in &rules {
        let (mut n, mut se, mut sr) = (0usize, 0f64, 0f64);
        for l in &mut model.layers {
            let r = apply_layer(l, rule.class, rule.scheme);
            n += r.0;
            se += r.1;
            sr += r.2;
        }
        let rms = if sr > 0.0 { (se / sr).sqrt() } else { 0.0 };
        let scheme = match rule.scheme {
            SimScheme::Nvfp4 => "nvfp4".to_string(),
            SimScheme::Int(b) => format!("int{b}"),
        };
        if n == 0 {
            eprintln!(
                "[qsim] WARNING: {:?} matched NO weights — this arch has none. Results are \
                 the UNMODIFIED baseline, not a measurement of {scheme}.",
                rule.class
            );
        } else {
            eprintln!(
                "[qsim] {:?} → {scheme}: {:.1}M weights, rms perturbation {:.5}",
                rule.class,
                n as f64 / 1e6,
                rms
            );
            // A perturbation far below the scheme's own reconstruction error means the
            // container's existing grid absorbed most of it — the arm is near-identity
            // and its perplexity will read as "free" no matter what the scheme costs.
            if rms < 0.01 {
                eprintln!(
                    "[qsim] WARNING: rms {rms:.5} is near zero — this arm barely changed the \
                     model. Do NOT read its perplexity as the cost of {scheme}."
                );
            }
        }
    }
    eprintln!(
        "[qsim] done in {:.1}s — VALUES carry the target error; bytes/token are UNCHANGED \
         (this measures quality, not speed)",
        t0.elapsed().as_secs_f64()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hybrid_spec_and_expands_mamba() {
        let r = parse_rules("mamba:nvfp4,shared:6").unwrap();
        // `mamba` expands to both projections, so 3 rules from 2 entries.
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].class, Class::MambaIn);
        assert_eq!(r[0].scheme, SimScheme::Nvfp4);
        assert_eq!(r[1].class, Class::MambaOut);
        assert_eq!(r[2].class, Class::Shared);
        assert_eq!(r[2].scheme, SimScheme::Int(6));
    }

    #[test]
    fn rejects_typos_instead_of_silently_doing_nothing() {
        // The failure mode this guards: a misspelled class silently simulating nothing
        // reads as "that precision is free" — the most expensive possible wrong answer.
        assert!(parse_rules("mmba:nvfp4").is_err(), "misspelled class must error");
        assert!(parse_rules("mamba:fp4").is_err(), "unknown scheme must error");
        assert!(parse_rules("mamba").is_err(), "missing scheme must error");
        assert!(parse_rules("mamba:99").is_err(), "out-of-range width must error");
        // ...and the valid form it is easy to typo *into* must still parse.
        assert!(parse_rules("mamba:nvfp4").is_ok());
    }

    #[test]
    fn refuses_formats_it_cannot_decode_instead_of_zeroing_them() {
        // `dequantize_qtensor` yields ZEROS for formats it does not decode (int2, nvfp4).
        // Simulating such a tensor would re-store those zeros — silently destroying the
        // weights while reporting success. Bail before the dequant instead.
        let w: Vec<f32> = (0..(4 * 32)).map(|k| ((k as f32) * 0.21).cos()).collect();
        let mut t2 = crate::quantize::qtensor_from_f32(&w, 4, 32, 2); // int2, fmt 3
        assert_eq!(t2.fmt_code, 3);
        let before = t2.clone();
        let r = simulate(&mut t2, SimScheme::Nvfp4);
        assert_eq!(r.0, 0, "int2 must be skipped, not simulated");
        assert_eq!(t2.fmt_code, before.fmt_code);
        assert_eq!(t2.q4, before.q4, "tensor must be left byte-identical");
    }

    #[test]
    fn simulating_int8_on_int8_is_near_identity_but_nvfp4_is_not() {
        // Sanity that the round trip actually bites: re-simulating the width a tensor is
        // already stored at must barely move it, while NVFP4 must move it a lot. Without
        // this, a broken `simulate` that returned the input unchanged would look fine.
        let w: Vec<f32> = (0..(16 * 64)).map(|k| ((k as f32) * 0.37).sin()).collect();
        let base = crate::quantize::qtensor_from_f32(&w, 16, 64, 8);
        let rms = |t: &QTensor| {
            let v = crate::convert::dequantize_qtensor_pub(t);
            let (mut se, mut sr) = (0f64, 0f64);
            for (a, b) in v.iter().zip(&w) {
                se += ((a - b) as f64).powi(2);
                sr += (*b as f64).powi(2);
            }
            (se / sr).sqrt()
        };
        let base_err = rms(&base);

        let mut t8 = base.clone();
        simulate(&mut t8, SimScheme::Int(8));
        let mut t4 = base.clone();
        simulate(&mut t4, SimScheme::Nvfp4);

        assert!(rms(&t8) < base_err * 1.5, "int8-on-int8 should be ~identity");
        assert!(rms(&t4) > base_err * 3.0, "nvfp4 must visibly degrade vs int8");
    }
}
