//! Throwaway: time the MXFP4 CPU matmul on one real K3 expert projection shape.
use colibri_core::{Bytes, QTensor};
fn main() {
    let (o, i) = (3072usize, 3584usize); // K3 expert gate/up: [moe_inter, moe_latent]
    let rb = i / 2;
    let nb = i / 32;
    let q4: Vec<u8> = (0..o * rb).map(|k| ((k * 7 + 3) % 256) as u8).collect();
    let bs: Vec<u8> = (0..o * nb).map(|k| (120 + (k % 12)) as u8).collect();
    let w = QTensor {
        fmt_code: 6,
        q4: Bytes::Owned(q4),
        bs: Bytes::Owned(bs),
        g: 1.0,
        o: o as i32,
        i: i as i32,
        ..Default::default()
    };
    let x: Vec<f32> = (0..i).map(|k| ((k % 13) as f32 - 6.0) * 0.05).collect();
    let mut y = vec![0f32; o];
    colibri_engine::matmul_qt(&mut y, &x, &w, 1);
    let t = std::time::Instant::now();
    const N: u32 = 5;
    for _ in 0..N {
        colibri_engine::matmul_qt(&mut y, &x, &w, 1);
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / N as f64;
    println!(
        "[mxbench] {o}x{i}  {ms:.1} ms/call  ({:.0} MMAC/s)  checksum {:.6}",
        (o * i) as f64 / (ms * 1e3),
        y[0]
    );
}
