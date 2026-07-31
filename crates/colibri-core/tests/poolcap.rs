//! The buffer pool is bounded by BYTES, not by a count — in its **own test binary**.
//!
//! Own process for the reason `arena.rs` spells out: the pool and its counters are
//! process-global, so a second `#[test]` beside this one would see the other's
//! allocations. That is not hypothetical — this assertion first lived in `hostpin.rs` and
//! immediately broke that test's page-lock counters.

use colibri_core::SharedBuf;

/// The byte cap governs; the count cap does not bind by default.
///
#[test]
fn the_pool_is_bounded_by_bytes_not_by_a_count() {
    // A count cap of 128 retained 128 MB for a model with 1 MB spans and 2.7 GB for one
    // with 21 MB spans, from the same number — the small-span model, which has the MOST
    // spans per batch, got a twentieth of the byte budget it was meant to have.
    //
    // 1 MiB is exactly K3's span size, and `POOL_MIN_BYTES`, so these are pooled.
    std::env::remove_var("COLI_BUF_POOL"); // default: unbounded count
    std::env::set_var("COLI_BUF_POOL_MB", "8");
    const SMALL: usize = 1 << 20;

    // Hold 200 at once — well past the old 128-entry cap — then release them all.
    let held: Vec<_> = (0..200).map(|_| SharedBuf::with_len(SMALL)).collect();
    drop(held);

    // With the count cap gone, retention is decided by the 8 MB byte cap alone, so the
    // pool keeps what fits and rejects the rest. The failure this guards is the opposite:
    // a count cap binding first would have rejected almost all 200 regardless of bytes.
    let (_, _, pushes, drops, _) = colibri_core::quant::pool_profile();
    assert!(
        pushes >= 8,
        "the byte cap should retain ~8 MiB worth, got {pushes} pushes"
    );
    assert!(
        pushes <= 9,
        "and no more than the byte cap allows, got {pushes}"
    );
    assert!(
        drops > 0,
        "the rest are rejected by the BYTE cap, which is the live bound"
    );
}
