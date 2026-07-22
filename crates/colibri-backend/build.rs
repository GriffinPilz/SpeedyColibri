//! Build script for the CUDA backend.
//!
//! When the `cuda` feature is enabled, compile the crate's `cuda/backend_cuda.cu`
//! with `nvcc` into a static lib and link it (plus `cudart` + `stdc++`), mirroring
//! the C Makefile's Linux CUDA path. When the feature is off — the default — this
//! is a no-op, so ordinary (CPU / non-CUDA) builds are unaffected.
//!
//! If `nvcc` is not found while the feature IS on we **fail the build**, naming the
//! cause. Silently skipping produced a link error (`undefined reference to
//! coli_cuda_*`) far from its cause — and worse, cargo *caches* build-script results,
//! so one build in a shell without `nvcc` poisoned every later build even after nvcc
//! was back on `PATH`. Hence `rerun-if-env-changed=PATH` below: `nvcc` is normally
//! found through it, so a PATH change must re-run the probe.
//!
//! `COLI_ALLOW_NO_NVCC=1` restores the old skip-with-warning, so `cargo check
//! --features cuda` can still type-check the FFI on a machine without CUDA.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Pick nvcc's `-arch`. `CUDA_ARCH` always wins; otherwise detect the local GPU.
///
/// **Why not `-arch=native`.** On a GB10 (compute 12.1) `native` resolves to the
/// *generic* target `sm_121`, and ptxas rejects block-scaled MMA there:
///
///   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale... (NVFP4)
///     sm_121   -> error: Instruction 'mma with block scale' not supported
///     sm_121a  -> compiles
///
/// The `a` suffix selects the *architecture-specific* target, which is where NVIDIA
/// exposes the sm_12x block-scaled tensor-core instructions. Verified by compiling the
/// instruction against each target on the box; FP8 (`m16n8k32...e4m3`) needs no suffix
/// and assembles for plain `sm_121` too.
///
/// The tradeoff: an arch-specific cubin carries no forward-JIT PTX, so it runs only on
/// this compute capability. `native` had the same practical scope (it emits for the
/// detected arch), and this project targets DGX Spark.
///
/// Falls back to `native` when no GPU can be queried — e.g. a GPU-less docker build,
/// which is why the Dockerfile pins CUDA_ARCH explicitly rather than relying on this.
fn detect_arch() -> String {
    if let Ok(a) = env::var("CUDA_ARCH") {
        return a;
    }
    let cap = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next().map(str::trim).map(str::to_string));
    let Some(cap) = cap.filter(|c| !c.is_empty()) else {
        return "native".to_string();
    };
    let mut parts = cap.split('.');
    match (
        parts.next().and_then(|v| v.parse::<u32>().ok()),
        parts.next().and_then(|v| v.parse::<u32>().ok()),
    ) {
        // Blackwell (12.x): take the arch-specific target so block-scaled MMA is
        // reachable. Only claimed for the family measured on this hardware.
        (Some(maj @ 12), Some(min)) => format!("sm_{maj}{min}a"),
        (Some(maj), Some(min)) => format!("sm_{maj}{min}"),
        _ => "native".to_string(),
    }
}

fn main() {
    // Only touch CUDA when the feature is enabled.
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // CUDA kernel source lives in the crate (`cuda/`). It is the live GPU backend,
    // not part of the ported-away C reference engine — no Rust equivalent exists.
    let cu = manifest.join("cuda/backend_cuda.cu");
    let hdr = manifest.join("cuda/backend_cuda.h");
    println!("cargo:rerun-if-changed={}", cu.display());
    println!("cargo:rerun-if-changed={}", hdr.display());
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    // `nvcc` is normally resolved through PATH, and cargo caches this script's
    // result. Without re-running on a PATH change, one build from a shell that
    // lacks nvcc (a non-interactive ssh, say) sticks — every later build reuses the
    // "no CUDA" outcome and fails at link with `undefined reference to coli_cuda_*`.
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-env-changed=COLI_ALLOW_NO_NVCC");

    let nvcc = env::var("NVCC").ok().unwrap_or_else(|| {
        env::var("CUDA_HOME")
            .map(|h| format!("{h}/bin/nvcc"))
            .unwrap_or_else(|_| "nvcc".to_string())
    });

    let have_nvcc = Command::new(&nvcc)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !have_nvcc {
        // Escape hatch: type-check the FFI on a machine without CUDA.
        if env::var_os("COLI_ALLOW_NO_NVCC").is_some() {
            println!(
                "cargo:warning=nvcc not found ('{nvcc}') and COLI_ALLOW_NO_NVCC is set: \
                 CUDA backend NOT compiled. Type-checking only — linking will fail with \
                 undefined `coli_cuda_*`."
            );
            return;
        }
        // Fail here, where the cause is known. Skipping would surface as a link error
        // against `coli_cuda_*` with no hint that nvcc was simply missing.
        panic!(
            "the `cuda` feature is enabled but nvcc was not found (tried '{nvcc}').\n\
             Build on a CUDA host, or point at it explicitly:\n\
             \x20   NVCC=/usr/local/cuda/bin/nvcc CUDA_HOME=/usr/local/cuda \
             cargo build --release -p coli --features cuda\n\
             A login shell (`bash -lc`) usually has nvcc on PATH where a plain \
             non-interactive ssh does not.\n\
             To type-check the FFI without CUDA, set COLI_ALLOW_NO_NVCC=1."
        );
    }

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lib = out.join("libcolibri_cuda.a");
    let arch = detect_arch();
    println!("cargo:warning=nvcc -arch={arch}");

    let status = Command::new(&nvcc)
        .args(["-O3", "-std=c++17"])
        .arg(format!("-arch={arch}"))
        .args(["-Xcompiler", "-fPIC"])
        .arg("-lib")
        .arg(&cu)
        .arg("-o")
        .arg(&lib)
        .status()
        .expect("failed to invoke nvcc");
    assert!(status.success(), "nvcc failed to compile backend_cuda.cu");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=colibri_cuda");
    if let Ok(home) = env::var("CUDA_HOME") {
        // `cudart` lives in `lib64` on x86 CUDA but under `targets/<arch>-linux/lib`
        // on the ARM/sbsa DGX Spark — add every candidate so the linker finds
        // `-lcudart` regardless of layout (missing it = `cannot find -lcudart` and a
        // silent push toward the CPU-only fallback). Only existing dirs are emitted.
        for sub in [
            "lib64",
            "lib",
            "targets/sbsa-linux/lib",   // ARM DGX Spark
            "targets/aarch64-linux/lib",
            "targets/x86_64-linux/lib",
        ] {
            let p = format!("{home}/{sub}");
            if std::path::Path::new(&p).is_dir() {
                println!("cargo:rustc-link-search=native={p}");
            }
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
