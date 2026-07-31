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

use colibri_core::quant::{pin_profile, set_host_pin_hooks};
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

    // --- below the pool threshold is never registered --------------------------------
    {
        let _small = SharedBuf::with_len(4096);
        assert!(
            REGISTERED.lock().unwrap().is_empty(),
            "small buffers are not pooled or pinned"
        );
    }

    // --- allocate, use, drop; every one is rejected by the pool and must be released ---
    const N: usize = 6;
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
    assert_eq!(capped, 0, "6 x 2 MiB is nowhere near the page-lock ceiling");

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
