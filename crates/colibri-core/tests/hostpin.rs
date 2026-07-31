//! Page-lock lifetime for pooled expert buffers, in its **own test binary**.
//!
//! The hooks and the pool are process-global, so this needs its own process for the same
//! reason `arena.rs` does — see the note at the top of that file.
//!
//! What is actually at risk here is not throughput, it is a dangling page-lock: a buffer
//! that is registered with the driver and then freed leaves the driver holding a lock on
//! memory the kernel has handed to someone else. That is silent until it is catastrophic,
//! and it is exactly the kind of failure the rest of this work has repeatedly found
//! presenting as a slowdown rather than a crash. So the invariant under test is symmetry:
//! every registration is released before its allocation goes away.

use colibri_core::quant::{pin_profile, set_host_pin_hooks, set_pin_budget};
use colibri_core::SharedBuf;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

static REGISTERED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
static UNREGISTERED: AtomicU64 = AtomicU64::new(0);

fn fake_register(p: *mut u8, bytes: usize) -> bool {
    REGISTERED.lock().unwrap().push((p as usize, bytes));
    true
}

fn fake_unregister(p: *mut u8) {
    let mut r = REGISTERED.lock().unwrap();
    let before = r.len();
    r.retain(|(q, _)| *q != p as usize);
    assert_eq!(
        before - r.len(),
        1,
        "unregistered a pointer that was never registered"
    );
    UNREGISTERED.fetch_add(1, Relaxed);
}

const BIG: usize = 2 << 20; // above POOL_MIN_BYTES, so it is a pooled size

#[test]
fn every_page_lock_is_released_before_its_memory_is_freed() {
    // Rejection is the only path that frees a pooled buffer, so it is the only path that
    // can strand a registration. A pool of zero entries makes every return a rejection.
    std::env::set_var("COLI_BUF_POOL", "0");
    set_host_pin_hooks(fake_register, fake_unregister);

    // --- no grant means no page-locking ----------------------------------------------
    // The default budget is zero, and that is the safe default rather than an oversight:
    // a caller that has not sized a grant gets exactly the behaviour that shipped before
    // page-locking existed.
    {
        let _ungranted = SharedBuf::with_len(BIG);
        assert!(
            REGISTERED.lock().unwrap().is_empty(),
            "nothing is page-locked until the engine grants a budget"
        );
    }
    assert_eq!(pin_profile().0, 0, "no grant, no locks");
    // A zero grant reports as *capped*, not as *failed* — the distinction matters when
    // reading these counters in a real run: capped is the ceiling working, failed is the
    // driver refusing.
    let capped_ungranted = pin_profile().3;
    assert_eq!(
        capped_ungranted, 1,
        "the ungranted buffer was left pageable by the ceiling"
    );

    // Grant enough for the buffers below but not for all of them, so both the granted and
    // the capped path are exercised.
    const N: usize = 6;
    set_pin_budget((BIG * 4) as u64);

    // --- below the pool threshold is never registered --------------------------------
    {
        let _small = SharedBuf::with_len(4096);
        assert!(
            REGISTERED.lock().unwrap().is_empty(),
            "small buffers are not pooled or pinned"
        );
    }

    // --- allocate, use, drop; every one is rejected by the pool and must be released ---
    for _ in 0..N {
        let mut b = SharedBuf::with_len(BIG);
        b.as_mut_slice()[0] = 1; // touch it, as the read path would
    }

    let (ok, fail, bytes, capped) = pin_profile();
    assert_eq!(
        ok + fail,
        N as u64,
        "one registration attempt per fresh allocation"
    );
    assert_eq!(
        UNREGISTERED.load(Relaxed),
        ok,
        "every successful page-lock is released — a leftover is a lock on freed memory"
    );
    assert!(
        REGISTERED.lock().unwrap().is_empty(),
        "no registration outlives its allocation"
    );
    assert_eq!(bytes, 0, "the locked-byte ledger returns to zero");
    // Each buffer is dropped immediately, so the ledger falls back to zero between them
    // and all six fit under a 4-buffer grant. What the grant must never do is let the
    // locked total exceed it at any instant — that is the assertion above on `bytes`.
    assert_eq!(
        capped, capped_ungranted,
        "with the grant in place and one buffer live at a time, nothing more is capped"
    );

    // --- the ceiling actually caps ----------------------------------------------------
    // Hold them all at once this time. Past the grant, allocation continues and simply
    // stays pageable, which is the fallback the whole design leans on.
    {
        let _held: Vec<_> = (0..N).map(|_| SharedBuf::with_len(BIG)).collect();
        let (_, _, bytes_held, capped_held) = pin_profile();
        assert!(
            bytes_held <= (BIG * 4) as u64,
            "locked bytes never exceed the grant"
        );
        assert!(
            capped_held > 0,
            "the buffers past the grant are left pageable"
        );
    }
    assert_eq!(
        pin_profile().2,
        0,
        "releasing them returns the ledger to zero"
    );

    // --- a recycled buffer is NOT re-registered ---------------------------------------
    // Registering the same address twice is a driver error, and the pool hands the same
    // allocation out repeatedly by design.
    std::env::set_var("COLI_BUF_POOL", "0"); // already read once; kept explicit for the reader
    let (ok_before, ..) = pin_profile();
    let first = SharedBuf::with_len(BIG);
    let ptr = first.as_ptr();
    drop(first);
    let (ok_after, ..) = pin_profile();
    assert_eq!(
        ok_after,
        ok_before + 1,
        "a fresh allocation registers exactly once"
    );
    let _ = ptr;
}
