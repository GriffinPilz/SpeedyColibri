//! Expert hot-store eviction policy — a faithful port of `c/tier.h`.
//!
//! The engine keeps a set of "pinned" experts resident in RAM/VRAM (the
//! hot-store) and, between turns, may swap a cold pinned slot for a hotter
//! unpinned expert. These functions decide *whether* and *what* to swap. The
//! hysteresis margins (fixed +4, plus 25%) exist to stop ping-ponging on tiny
//! samples.

/// A chosen swap: replace hot-store `slot` with expert `eid`, expecting `gain`
/// heat units of improvement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Swap {
    pub slot: usize,
    pub eid: usize,
    pub gain: i64,
}

/// Pure-frequency swap pick — port of `tier_pick_swap`.
///
/// Picks the coldest pinned slot and the hottest unpinned expert; returns a
/// `Swap` only if the candidate beats the cold slot by the hysteresis margin.
pub fn pick_swap(heat: &[u32], pinned: &[usize]) -> Option<Swap> {
    let nexpert = heat.len();
    if pinned.is_empty() || nexpert < 1 {
        return None;
    }
    // coldest pinned slot
    let mut cold = 0usize;
    for z in 1..pinned.len() {
        if heat[pinned[z]] < heat[pinned[cold]] {
            cold = z;
        }
    }
    // hottest non-resident expert
    let mut hot: isize = -1;
    let mut fh: u32 = 0;
    for (e, &h) in heat.iter().enumerate() {
        let resident = pinned.iter().any(|&p| p == e);
        if !resident && h > fh {
            fh = h;
            hot = e as isize;
        }
    }
    if hot < 0 {
        return None;
    }
    let fc = heat[pinned[cold]];
    // fh must exceed fc by 25% + 4 to be worth swapping.
    if fh <= fc + (fc >> 2) + 4 {
        return None;
    }
    Some(Swap {
        slot: cold,
        eid: hot as usize,
        gain: fh as i64 - fc as i64,
    })
}

/// LFRU score — frequency is primary, recency breaks near-ties.
///
/// A recent access is worth at most 255 points; one frequency count is worth
/// 256, so a merely-recent expert can never displace a genuinely hotter one.
pub fn lfru_score(heat: u32, last: u32, clock: u32) -> u64 {
    let age = clock.wrapping_sub(last);
    let recent = if age < 255 { 255 - age } else { 0 };
    ((heat as u64) << 8) | recent as u64
}

/// LFRU swap pick — port of `tier_pick_lfru`.
pub fn pick_lfru(heat: &[u32], last: &[u32], clock: u32, pinned: &[usize]) -> Option<Swap> {
    let nexpert = heat.len();
    if last.len() < nexpert || pinned.is_empty() || nexpert < 1 {
        return None;
    }
    let score = |e: usize| lfru_score(heat[e], last[e], clock);

    let mut cold = 0usize;
    for z in 1..pinned.len() {
        if score(pinned[z]) < score(pinned[cold]) {
            cold = z;
        }
    }
    let mut hot: isize = -1;
    let mut hs: u64 = 0;
    for e in 0..nexpert {
        let resident = pinned.iter().any(|&p| p == e);
        let sc = score(e);
        if !resident && (hot < 0 || sc > hs) {
            hot = e as isize;
            hs = sc;
        }
    }
    if hot < 0 {
        return None;
    }
    let cs = score(pinned[cold]);
    // same 25% + 4-frequency hysteresis, expressed in score units.
    if hs <= cs + (cs >> 2) + (4u64 << 8) {
        return None;
    }
    Some(Swap {
        slot: cold,
        eid: hot as usize,
        gain: ((hs - cs) >> 8) as i64,
    })
}

/// Halve every heat counter — port of `tier_decay`.
pub fn decay(heat: &mut [u32]) {
    for h in heat.iter_mut() {
        *h >>= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_swap_when_margin_not_cleared() {
        // pinned {0,1} with heat 100,100; candidate expert 2 heat 110 — below
        // the 25%+4 margin over 100, so no swap.
        let heat = [100, 100, 110];
        assert_eq!(pick_swap(&heat, &[0, 1]), None);
    }

    #[test]
    fn swaps_hottest_for_coldest() {
        // pinned {0,1} heat 10,200; expert 2 heat 500 clears the margin over the
        // cold slot (slot 0, heat 10).
        let heat = [10, 200, 500];
        let s = pick_swap(&heat, &[0, 1]).unwrap();
        assert_eq!((s.slot, s.eid, s.gain), (0, 2, 490));
    }

    #[test]
    fn lfru_frequency_dominates_recency() {
        // Expert A: high freq, old. Expert B: low freq, brand new.
        // A must outscore B despite being less recent.
        let a = lfru_score(100, 0, 1000); // very old
        let b = lfru_score(1, 1000, 1000); // just accessed
        assert!(a > b);
    }

    #[test]
    fn decay_halves() {
        let mut h = [8, 3, 0, 255];
        decay(&mut h);
        assert_eq!(h, [4, 1, 0, 127]);
    }
}
