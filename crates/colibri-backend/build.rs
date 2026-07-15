//! Build script for the CUDA backend.
//!
//! When the `cuda` feature is enabled, compile the reference `c/backend_cuda.cu`
//! with `nvcc` into a static lib and link it (plus `cudart` + `stdc++`), mirroring
//! the C Makefile's Linux CUDA path. When the feature is off — the default — this
//! is a no-op, so ordinary (CPU / non-CUDA) builds are unaffected.
//!
//! If `nvcc` is not found while the feature IS on, we skip compilation with a
//! warning rather than aborting, so `cargo check --features cuda` can still
//! type-check the FFI on a machine without CUDA. A real `cargo build
//! --features cuda` there will fail at link time (undefined `coli_cuda_*`), as
//! expected — build on a CUDA host (e.g. the DGX Spark image's `-devel` base).

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only touch CUDA when the feature is enabled.
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // crates/colibri-backend -> repo root c/
    let cu = manifest.join("../../c/backend_cuda.cu");
    let hdr = manifest.join("../../c/backend_cuda.h");
    println!("cargo:rerun-if-changed={}", cu.display());
    println!("cargo:rerun-if-changed={}", hdr.display());
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

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
        println!(
            "cargo:warning=nvcc not found ('{nvcc}'); CUDA backend NOT compiled. \
             `cargo check --features cuda` still type-checks the FFI, but linking \
             will fail. Build on a CUDA host or set NVCC / CUDA_HOME."
        );
        return;
    }

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lib = out.join("libcolibri_cuda.a");
    // `native` matches the C Makefile default; set CUDA_ARCH=sm_121 for GB10.
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "native".to_string());

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
        println!("cargo:rustc-link-search=native={home}/lib64");
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
