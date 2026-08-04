//! Hyper-Connections: DeepSeek-V4's replacement for the residual stream.
//!
//! A normal block is `x = x + f(norm(x))`. Under Hyper-Connections the inter-block state is
//! `hc_mult` copies of the hidden state, `[hc, d]`, and each block runs
//!
//! ```text
//!   residual = x                          // [hc, d]
//!   (x, post, comb) = hc_pre(x)           // collapse [hc,d] -> [d]
//!   x = mixer(norm(x))                    // ordinary [d] -> [d]
//!   x = hc_post(x, residual, post, comb)  // expand [d] -> [hc,d]
//! ```
//!
//! twice per block — once around attention, once around the FFN — and one `hc_head` collapse
//! at the very end before the LM head.
//!
//! **The mixer never sees the extra axis.** `hc_pre` collapses before `attn_norm` and
//! `hc_post` expands after, so attention, the MoE and every scratch buffer keep their usual
//! `[d]` shape. Only the carried residual is `hc`-wide. Nothing else in the engine should be
//! widened for this — at a 4096-token prefill it is +201 MB, against a ~99 GB expert budget.
//!
//! Ported from the checkpoint's own `inference/kernel.py` (`hc_split_sinkhorn_kernel`) and
//! `inference/model.py` (`Block.hc_pre` / `hc_post` / `hc_head`). The tests check against
//! vectors produced by a transcription of that source, not against a re-derivation.

/// Largest `hc_mult` this supports on the stack. DeepSeek-V4 uses 4; the Sinkhorn works on
/// an `hc x hc` matrix, so keeping it fixed-size avoids a heap allocation per token per
/// block — and there are two HC wraps on each of 43 layers, so that is 86 per token.
pub const MAX_HC: usize = 8;

/// `(2 + hc) * hc` — the width of the mix projection: `hc` pre-weights, `hc` post-weights,
/// then an `hc x hc` combination matrix, concatenated in that order.
pub const fn mix_width(hc: usize) -> usize {
    (2 + hc) * hc
}

/// Split the raw mix vector into `pre`, `post` and a Sinkhorn-normalised `comb`.
///
/// `comb` is made doubly stochastic: one row softmax, then a column normalisation, then
/// `iters - 1` further row/column rounds. The reference does exactly this many, and the
/// count matters — it is `hc_sinkhorn_iters` (20) from the config, not a convergence loop,
/// so the result is reproducible rather than tolerance-dependent.
fn split_sinkhorn(
    mixes: &[f32],
    scale: &[f32],
    base: &[f32],
    hc: usize,
    iters: usize,
    eps: f32,
    pre: &mut [f32; MAX_HC],
    post: &mut [f32; MAX_HC],
    comb: &mut [[f32; MAX_HC]; MAX_HC],
) {
    let sig = |z: f32| 1.0 / (1.0 + (-z).exp());
    for j in 0..hc {
        pre[j] = sig(mixes[j] * scale[0] + base[j]) + eps;
        post[j] = 2.0 * sig(mixes[j + hc] * scale[1] + base[j + hc]);
    }
    for j in 0..hc {
        for k in 0..hc {
            let i = j * hc + k + 2 * hc;
            comb[j][k] = mixes[i] * scale[2] + base[i];
        }
    }
    // Row softmax (+eps), matching the kernel's max-subtract form.
    for row in comb.iter_mut().take(hc) {
        let mut m = f32::NEG_INFINITY;
        for &v in row.iter().take(hc) {
            m = m.max(v);
        }
        let mut s = 0.0;
        for v in row.iter_mut().take(hc) {
            *v = (*v - m).exp();
            s += *v;
        }
        for v in row.iter_mut().take(hc) {
            *v = *v / s + eps;
        }
    }
    // One column normalisation, then iters-1 row/column rounds.
    let mut colnorm = |comb: &mut [[f32; MAX_HC]; MAX_HC]| {
        for k in 0..hc {
            let mut s = 0.0;
            for row in comb.iter().take(hc) {
                s += row[k];
            }
            let inv = 1.0 / (s + eps);
            for row in comb.iter_mut().take(hc) {
                row[k] *= inv;
            }
        }
    };
    colnorm(comb);
    for _ in 0..iters.saturating_sub(1) {
        for row in comb.iter_mut().take(hc) {
            let s: f32 = row.iter().take(hc).sum();
            let inv = 1.0 / (s + eps);
            for v in row.iter_mut().take(hc) {
                *v *= inv;
            }
        }
        colnorm(comb);
    }
}

/// Collapse `[hc, d]` -> `[d]`, returning the `post`/`comb` weights the matching
/// [`hc_post`] needs. `x` is row-major `[hc][d]`.
///
/// The RMS is taken over the WHOLE flattened `hc*d` vector, not per copy — the reference
/// flattens first and that is what the projection is normalised against.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre(
    x: &[f32],
    hc_fn: &[f32],
    scale: &[f32],
    base: &[f32],
    hc: usize,
    d: usize,
    norm_eps: f32,
    eps: f32,
    iters: usize,
    out: &mut [f32],
    post: &mut [f32; MAX_HC],
    comb: &mut [[f32; MAX_HC]; MAX_HC],
) {
    debug_assert_eq!(x.len(), hc * d);
    debug_assert_eq!(out.len(), d);
    debug_assert!(hc <= MAX_HC);
    let n = hc * d;
    let mut ss = 0.0f64;
    for &v in x {
        ss += (v as f64) * (v as f64);
    }
    let rsqrt = (1.0 / ((ss / n as f64) + norm_eps as f64).sqrt()) as f32;

    let mw = mix_width(hc);
    let mut mixes = [0.0f32; mix_width(MAX_HC)];
    for (m, row) in mixes.iter_mut().take(mw).zip(hc_fn.chunks_exact(n)) {
        let mut acc = 0.0f64;
        for (a, b) in row.iter().zip(x) {
            acc += (*a as f64) * (*b as f64);
        }
        *m = (acc as f32) * rsqrt;
    }

    let mut pre = [0.0f32; MAX_HC];
    split_sinkhorn(
        &mixes, scale, base, hc, iters, eps, &mut pre, post, comb,
    );
    out.fill(0.0);
    for i in 0..hc {
        let w = pre[i];
        let src = &x[i * d..(i + 1) * d];
        for (o, &v) in out.iter_mut().zip(src) {
            *o += w * v;
        }
    }
}

/// Expand `[d]` -> `[hc, d]`, mixing the previous copies through `comb`.
///
/// `out[k] = post[k] * x + sum_j comb[j][k] * residual[j]` — note the index order: `comb`
/// is summed over its FIRST index, so it is effectively transposed here. Getting that
/// backwards is silent (both are `hc x hc`) and would corrupt the residual stream, which is
/// why the test pins a non-symmetric `comb`.
pub fn hc_post(
    x: &[f32],
    residual: &[f32],
    post: &[f32; MAX_HC],
    comb: &[[f32; MAX_HC]; MAX_HC],
    hc: usize,
    d: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), d);
    debug_assert_eq!(residual.len(), hc * d);
    debug_assert_eq!(out.len(), hc * d);
    for k in 0..hc {
        let dst = &mut out[k * d..(k + 1) * d];
        let p = post[k];
        for (o, &v) in dst.iter_mut().zip(x) {
            *o = p * v;
        }
        for j in 0..hc {
            let c = comb[j][k];
            if c == 0.0 {
                continue;
            }
            let src = &residual[j * d..(j + 1) * d];
            for (o, &v) in dst.iter_mut().zip(src) {
                *o += c * v;
            }
        }
    }
}

/// Final collapse before the LM head. Same shape as [`hc_pre`] but with **no Sinkhorn** and
/// no post/comb: the reference uses a plain sigmoid gate, and `hc_head_scale` is a single
/// scalar rather than the 3 that `hc_pre` takes.
#[allow(clippy::too_many_arguments)]
pub fn hc_head(
    x: &[f32],
    hc_fn: &[f32],
    scale: f32,
    base: &[f32],
    hc: usize,
    d: usize,
    norm_eps: f32,
    eps: f32,
    out: &mut [f32],
) {
    let n = hc * d;
    let mut ss = 0.0f64;
    for &v in x {
        ss += (v as f64) * (v as f64);
    }
    let rsqrt = (1.0 / ((ss / n as f64) + norm_eps as f64).sqrt()) as f32;
    out.fill(0.0);
    for (i, row) in hc_fn.chunks_exact(n).take(hc).enumerate() {
        let mut acc = 0.0f64;
        for (a, b) in row.iter().zip(x) {
            acc += (*a as f64) * (*b as f64);
        }
        let m = (acc as f32) * rsqrt;
        let w = 1.0 / (1.0 + (-(m * scale + base[i])).exp()) + eps;
        let src = &x[i * d..(i + 1) * d];
        for (o, &v) in out.iter_mut().zip(src) {
            *o += w * v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vectors from a transcription of the checkpoint's own inference/kernel.py and
    // model.py (see scratch hc_ref.py). hc=4, d=6. Checking against NUMBERS rather than
    // against a re-derivation of the same source is the point: a re-derivation would
    // reproduce my own misreading.
    const HC: usize = 4;
    const D: usize = 6;
    const X: [f32; 24] = [-0.0470720827f32, -0.3346437724f32, 0.3829738126f32, 0.2710156566f32, 0.2691233924f32, 0.318204797f32, -0.4711833858f32, -0.4324879589f32, -0.4990860013f32, -0.7421357736f32, 0.7305681343f32, 0.3723367578f32, -0.1907859377f32, 0.180488907f32, 0.2311154435f32, -0.1639313612f32, -0.2332155447f32, -0.4784615371f32, 0.3887652046f32, -0.1540673972f32, -0.8052667841f32, 0.3940061826f32, 0.8127646348f32, -0.3696993947f32];
    const BASE: [f32; 24] = [-0.1286663043f32, 0.3890315682f32, 0.1976511635f32, -0.3609311648f32, 0.2082704797f32, -0.4820786696f32, 0.0538849064f32, -0.3670571418f32, 0.2731799675f32, 0.0346873696f32, -0.591144389f32, -0.2953774563f32, -0.5886357607f32, 0.1341712151f32, -0.1145900374f32, 0.0877599472f32, -0.0435194036f32, 0.3845365822f32, -0.2380177277f32, 0.0519907573f32, -0.0560018911f32, 0.1469145461f32, -0.468412322f32, -0.4277506204f32];
    const SCALE: [f32; 3] = [0.7f32, 1.1f32, 0.9f32];
    const MIXOUT: [f32; 6] = [-0.0217522286f32, -0.2929134998f32, 0.3609523466f32, -0.3931969563f32, -0.6374767752f32, 0.2398736142f32];
    const WANT_PRE_Y: [f32; 6] = [-0.2837687326f32, -0.4159690495f32, -0.4232919965f32, -0.3149916999f32, 0.8768572811f32, -0.0622871655f32];
    const WANT_POST: [f32; 4] = [1.292768066f32, 1.265299316f32, 1.086667026f32, 1.248078649f32];
    const WANT_COMB: [f32; 16] = [0.2637811321f32, 0.241773518f32, 0.1536398982f32, 0.3408044516f32, 0.0939242598f32, 0.2137637196f32, 0.3700485206f32, 0.3222625f32, 0.2293010129f32, 0.3418954601f32, 0.2758348374f32, 0.1529676896f32, 0.4129925952f32, 0.2025663023f32, 0.2004757438f32, 0.1839643587f32];
    const WANT_OUT: [f32; 24] = [0.0320168775f32, -0.5298054481f32, 0.2411984359f32, -0.3813961965f32, -0.402312658f32, 0.1666139967f32, -0.1261038921f32, -0.5134820127f32, 0.3585166092f32, -0.5668641471f32, -0.5004596655f32, 0.2215650366f32, -0.179917709f32, -0.5108572538f32, 0.1687022675f32, -0.6264908921f32, -0.2824206206f32, 0.2412422445f32, -0.1527007459f32, -0.6197357621f32, 0.3073922027f32, -0.5901330204f32, -0.3546227069f32, 0.3866162541f32];

    fn hc_fn() -> Vec<f32> {
        // [mix_width(4)=24, hc*d=24], generated with the same seed as the reference.
        let mut v = Vec::with_capacity(24 * 24);
        v.extend_from_slice(&[-0.1419604177f32, 0.1582123495f32, 0.07478858203f32, 0.07593068719f32, 0.05184937939f32, 0.04648224098f32, -0.0275597017f32, -0.1451119403f32, 0.1462586058f32, -0.1095756518f32, 0.1384017243f32, -0.09343117214f32, 0.08500677987f32, -0.08823111942f32, -0.1247331388f32, 0.2089775942f32, 0.0585819861f32, 0.06614534257f32, -0.02940268673f32, -0.04369934974f32, 0.1387951841f32, 0.03127508652f32, -0.02876266045f32, -0.05782161149f32]);v.extend_from_slice(&[-0.01617434566f32, -0.1202506119f32, 0.1183780302f32, 0.05149274395f32, 0.181795877f32, -0.1656533252f32, 0.09733130785f32, -0.242748845f32, -0.1055257661f32, 0.0271150713f32, -0.08875351888f32, 0.1298912895f32, 0.02779062148f32, 0.1171839481f32, 0.1189793645f32, 0.1167781178f32, -0.0157995526f32, -0.008880130708f32, -0.05642145211f32, 0.02337011906f32, 0.006924760897f32, 0.1491965365f32, 0.1241362979f32, 0.09455440445f32]);v.extend_from_slice(&[0.05396419379f32, 0.02304019955f32, 0.05492527527f32, -0.05328136417f32, 0.08145236415f32, 0.08471602317f32, 0.2604218471f32, -0.01368202339f32, 0.001493490134f32, 0.05581268038f32, 0.04475185485f32, -0.03912116975f32, 0.2046495243f32, -0.06903693744f32, 0.1424865127f32, 0.1511700245f32, 0.07904187886f32, -0.16991326f32, 0.1096196112f32, -0.08349710941f32, -0.1187127705f32, 0.04095976467f32, 0.08470305333f32, 0.04003909684f32]);v.extend_from_slice(&[-0.08153930781f32, 0.03014733638f32, 0.1907607197f32, -0.05837230549f32, 0.05010138453f32, 0.0423919501f32, -0.0769755351f32, -0.06034970226f32, 0.01677458288f32, -0.08431077576f32, -0.002712022509f32, -0.05638040015f32, 0.05952080708f32, -0.0575060811f32, -0.05440269542f32, -0.09559834024f32, 0.1194028786f32, 0.1185537005f32, -0.01072693603f32, 0.008466346985f32, -0.07588380209f32, -0.005127093445f32, -0.008485717489f32, -0.1215269507f32]);v.extend_from_slice(&[-0.1501626248f32, 0.05193869356f32, 0.001107531871f32, 0.1603403013f32, -8.668298635e-05f32, -0.07670554111f32, -0.01408408755f32, 0.0897512645f32, -0.0751834972f32, -0.08858557411f32, -0.06417307591f32, -0.1002862999f32, -0.05067418523f32, 0.1048167327f32, -0.07653057235f32, -0.1392514541f32, -0.07332791425f32, 0.007677362108f32, 0.05402062916f32, -0.05416289829f32, -0.1682617656f32, 0.03915956744f32, -0.05069432939f32, 0.067852202f32]);v.extend_from_slice(&[-0.07765951057f32, -0.1790569727f32, -0.04396967167f32, -0.02607972636f32, 0.06259945164f32, -0.1216313062f32, -0.003629600862f32, 0.04095092412f32, -0.1293823793f32, -0.1068339112f32, 0.0962780163f32, 0.01956936475f32, 0.000918832836f32, -0.01010076646f32, 0.08406808478f32, 0.03458546166f32, -0.1879850555f32, 0.1797412561f32, -0.1719106844f32, 0.158355536f32, -0.05980725656f32, 0.1706796734f32, 0.1881303749f32, -0.103151091f32]);v.extend_from_slice(&[0.05938564813f32, 0.1609665674f32, 0.10164078f32, -0.1090515022f32, 0.07851035904f32, -0.07445025691f32, 0.03792505595f32, -0.1087395437f32, 0.04312088924f32, 0.1146157546f32, -0.1852109859f32, -0.001239578576f32, -0.1600072919f32, 0.02075977164f32, 0.0404652344f32, 0.3007578234f32, -0.09061888419f32, -0.02626030054f32, -0.04935211405f32, -0.08212701346f32, -0.1296410475f32, -0.1087692619f32, 0.2515904742f32, -0.06285244759f32]);v.extend_from_slice(&[0.05237462974f32, -0.08090594408f32, -0.06561852089f32, 0.05613188217f32, 0.0388214405f32, 0.1353736947f32, -0.01748150461f32, -0.07551239513f32, -0.07359708788f32, -0.1424900181f32, -0.03653626312f32, 0.1092916634f32, 0.07941623188f32, 0.08764078521f32, 0.02012031041f32, 0.08646096691f32, -0.04198053621f32, 0.008452092361f32, -0.2240122455f32, -0.02741686345f32, -0.1046275005f32, -0.02664232059f32, 0.1196438159f32, 0.007940644598f32]);v.extend_from_slice(&[-0.07257462886f32, -0.02594132214f32, 0.05078250919f32, 0.07460927103f32, 0.03493487593f32, 0.04899695744f32, 0.05568259794f32, 0.0198970388f32, 0.04475996578f32, -0.05514405296f32, 0.06182306526f32, -0.1462197441f32, 0.01684543943f32, -0.1260644707f32, -0.1224180267f32, 0.09720174702f32, 0.08534999305f32, -0.09267960988f32, 0.1072765617f32, 0.02585864934f32, 0.2191520342f32, -0.07019736778f32, -0.08058719436f32, -0.03784841056f32]);v.extend_from_slice(&[0.1019782451f32, -0.04751224494f32, -0.0439076457f32, -0.02327692911f32, -0.07705091474f32, 0.02774686326f32, -0.06152058051f32, -0.1316708658f32, -0.1447354088f32, 0.2144026632f32, -0.01505952586f32, 0.005630665742f32, -0.1678654522f32, 0.1285199561f32, 0.02241448061f32, -0.1150802292f32, 0.08700858178f32, 0.0377852586f32, 0.06522184768f32, 0.04135610806f32, 0.05638505353f32, 0.008066544593f32, 0.01535751573f32, -0.06639576335f32]);v.extend_from_slice(&[0.1750061056f32, 0.2347703318f32, 0.0715364329f32, 0.05195499228f32, -0.04648330628f32, -0.1204496521f32, 0.06690907029f32, 0.2230135162f32, -0.1189799173f32, 0.1352117774f32, 0.0522119112f32, 0.1452517539f32, 0.08494538338f32, -0.09346013327f32, -0.002288552206f32, 0.08469781151f32, 0.04626327696f32, 0.0995359452f32, -0.2181358055f32, 0.1815346377f32, -0.1360768087f32, 0.1857449116f32, -0.133869119f32, 0.01312072051f32]);v.extend_from_slice(&[-0.003602099299f32, -0.1419261285f32, -0.02634044683f32, 0.06106023606f32, 0.2566913562f32, 0.1436775811f32, -0.06388819928f32, -0.1070495791f32, -0.08131418303f32, -0.02348719282f32, 0.06025821893f32, -0.07212842341f32, 0.02747083083f32, -0.1068637544f32, -0.06072085665f32, -0.008569510232f32, -0.006156952142f32, 0.09762076065f32, -0.07191130339f32, 0.2174304482f32, -0.01538642714f32, 0.08300453771f32, -0.06833907494f32, 0.02113558871f32]);v.extend_from_slice(&[0.0290456556f32, 0.1764418806f32, 0.1590756568f32, 0.05706166168f32, 0.01798977436f32, 0.006316260875f32, -0.04608475779f32, -0.05785195101f32, 0.2192366582f32, -0.02055309319f32, -0.02919566945f32, -0.01735397583f32, 0.05636444391f32, -0.1531578201f32, -0.1180336318f32, -0.05045473086f32, -0.07950869503f32, 0.2612472443f32, 0.01576248478f32, -0.09550374651f32, -0.02858789883f32, 0.01388412455f32, 0.127876909f32, 0.1030330405f32]);v.extend_from_slice(&[-0.02360818665f32, -0.1510965785f32, -0.008120583851f32, 0.08802135692f32, 0.03898761691f32, 0.0109804617f32, 0.0541632103f32, 0.1249036896f32, 0.0185128907f32, 0.003852346005f32, 0.2401907953f32, -0.09621934134f32, 0.2201856864f32, -0.01735897891f32, -0.1328962456f32, 0.036396747f32, -0.07202660088f32, -0.09070029551f32, 0.1082890646f32, -0.01354479103f32, 0.1455405326f32, -0.1179315406f32, 0.1010200992f32, -0.08565196941f32]);v.extend_from_slice(&[0.1273183243f32, 0.1154980883f32, 0.193120941f32, 0.01640369201f32, 0.01633115137f32, -0.03675396404f32, -0.1199874063f32, 0.02348364829f32, 0.1342657867f32, -0.04041599697f32, 0.1410102163f32, 0.03131445521f32, 0.05505930553f32, -0.1124321329f32, -0.02538926869f32, -0.02414722862f32, -0.161955768f32, 0.2311845609f32, 0.06729457924f32, 0.1814858715f32, -0.01916586862f32, 0.0913001589f32, 0.05827955131f32, 0.2006919911f32]);v.extend_from_slice(&[-0.003035492176f32, 0.04011322183f32, -0.006656150491f32, 0.07943209588f32, 0.01790093244f32, 0.07383257698f32, -0.03317225449f32, 0.01838084935f32, 0.04226023386f32, 0.04762988442f32, 0.02086171679f32, 0.04812332216f32, 0.002039312455f32, 0.05561486748f32, -0.1887348009f32, 0.04405159464f32, -0.09734606868f32, -0.06775590866f32, -0.07535760809f32, 0.08370010945f32, -0.1132149111f32, 0.003374904942f32, 0.01365355552f32, -0.123145628f32]);v.extend_from_slice(&[0.03391371727f32, -0.1726650806f32, 0.05003976948f32, 0.06461991849f32, 0.03849056867f32, -0.06542845101f32, -0.004568101906f32, 0.09708411289f32, -0.007123301375f32, 0.003009580643f32, 0.01802104941f32, -0.1355710291f32, -0.2183064927f32, 0.04210369318f32, -0.01730192915f32, -0.03637413104f32, -0.02820894695f32, 0.08639707002f32, 0.03473536952f32, 0.08064145738f32, -0.16388813f32, 0.007705235366f32, -0.07648956908f32, 0.01332652053f32]);v.extend_from_slice(&[0.02152103553f32, -0.1011559073f32, 0.01115978387f32, -0.154831466f32, 0.1172056757f32, -0.003598962067f32, 0.001360554203f32, 0.08739054724f32, 0.03770305479f32, -0.05954008364f32, -0.143783524f32, 0.2363633829f32, -0.09683041834f32, -0.09307841494f32, 0.04434035307f32, -0.08769840428f32, -0.07772907357f32, -0.03284558068f32, 0.1301306337f32, -0.05137907482f32, -0.1898079676f32, 0.185853876f32, -0.1170202096f32, 0.01947904231f32]);v.extend_from_slice(&[0.02807484045f32, 0.03805075177f32, -0.02351078491f32, -0.09337960856f32, 0.01218764541f32, -0.03395527801f32, 0.02337004314f32, -0.08779965538f32, 0.08478435353f32, -0.08624419454f32, 0.01082216362f32, 0.01969371541f32, -0.1026966921f32, 0.04124895644f32, -0.01813941893f32, -0.1732806956f32, -0.06323260341f32, -0.004050501679f32, 0.1581679591f32, 0.02151736876f32, 0.08504904061f32, -0.1392313701f32, -0.01592023462f32, -0.0302185795f32]);v.extend_from_slice(&[-0.1434685135f32, 0.01258847379f32, -0.08106131747f32, -0.01658278857f32, -0.03062468323f32, -0.0448566825f32, 0.01582863465f32, -0.03688517352f32, -0.08889426503f32, 0.2252754349f32, 0.09120428466f32, 0.1030048728f32, 0.1723118822f32, 0.04383516002f32, 0.1167259855f32, 0.2522866323f32, 0.05797719628f32, -0.1520939923f32, 0.2580476832f32, -0.0853754126f32, 0.06587944904f32, -0.1057535782f32, -0.1315304535f32, 0.1311387386f32]);v.extend_from_slice(&[-0.008329995471f32, -0.1200111415f32, 0.0370187512f32, 0.04140362805f32, 0.1020108946f32, 0.1399876333f32, 0.1619042256f32, 0.2310610409f32, 0.1194114751f32, -0.2144655138f32, -0.1099121973f32, 0.05128459614f32, -0.0128484769f32, -0.01680138965f32, 0.2013157354f32, -0.04891148389f32, -0.1288795561f32, -0.07438963037f32, 0.09718541262f32, -0.05912130409f32, -0.01014013637f32, 0.1387613515f32, 0.1154752866f32, 0.1644429227f32]);v.extend_from_slice(&[-0.03839642754f32, -0.04007509585f32, -0.04335319352f32, 0.198211611f32, -0.01427604986f32, -0.007689741812f32, -0.01917207732f32, -0.002163708499f32, -0.008788845053f32, -0.1210492173f32, -0.05750217036f32, -0.04354982481f32, -0.0310894246f32, -0.1275665588f32, 0.1806201943f32, -0.0547434121f32, 0.1558412868f32, 0.1487460454f32, -0.1085182187f32, -0.1015298014f32, -0.04650932857f32, 0.1039908552f32, -0.1276563164f32, -0.0293799094f32]);v.extend_from_slice(&[-0.1247308074f32, -0.034429113f32, -0.01938741883f32, -0.09278125355f32, -0.123995429f32, -0.1409212443f32, 0.04611389469f32, -0.01481240008f32, -0.131931852f32, 0.04987385307f32, -0.0213264404f32, -0.05472454496f32, -0.004443546464f32, 0.002418278766f32, 0.005618987668f32, -0.01703680785f32, 0.1568060207f32, 0.02017345886f32, 0.09106220129f32, -0.01961598348f32, 0.01691815066f32, 0.05582800566f32, -0.04393242972f32, 0.02786551559f32]);v.extend_from_slice(&[-0.01863346787f32, 0.1039198507f32, -0.05962486453f32, -0.03706364083f32, -0.005268391121f32, -0.01586348182f32, -0.0229256374f32, -0.0008008524515f32, -0.01954980198f32, -0.02313769935f32, -0.1386161641f32, 0.04855621188f32, 0.01893778802f32, -0.03224850847f32, 0.04135410015f32, 0.1256150204f32, -0.05959402462f32, -0.1909852139f32, -0.1531132847f32, -0.1639883566f32, -0.04989504257f32, 0.192070272f32, -0.0565187564f32, -0.04762427355f32]);
        v
    }

    #[test]
    fn hc_matches_the_reference_kernel() {
        let f = hc_fn();
        let mut y = vec![0f32; D];
        let mut post = [0f32; MAX_HC];
        let mut comb = [[0f32; MAX_HC]; MAX_HC];
        hc_pre(&X, &f, &SCALE, &BASE, HC, D, 1e-6, 1e-6, 20, &mut y, &mut post, &mut comb);

        for (g, w) in y.iter().zip(WANT_PRE_Y.iter()) {
            assert!((g - w).abs() < 2e-5, "hc_pre: got {g} want {w}");
        }
        for (j, w) in WANT_POST.iter().enumerate() {
            assert!((post[j] - w).abs() < 2e-5, "post[{j}]: got {} want {w}", post[j]);
        }
        for j in 0..HC {
            for k in 0..HC {
                let w = WANT_COMB[j * HC + k];
                assert!((comb[j][k] - w).abs() < 2e-5,
                        "comb[{j}][{k}]: got {} want {w}", comb[j][k]);
            }
        }

        // Sinkhorn makes comb doubly stochastic — the property the 20 iterations exist for.
        for j in 0..HC {
            let r: f32 = (0..HC).map(|k| comb[j][k]).sum();
            let c: f32 = (0..HC).map(|k| comb[k][j]).sum();
            assert!((r - 1.0).abs() < 1e-3, "row {j} sums to {r}");
            assert!((c - 1.0).abs() < 1e-3, "col {j} sums to {c}");
        }

        let mut out = vec![0f32; HC * D];
        hc_post(&MIXOUT, &X, &post, &comb, HC, D, &mut out);
        for (i, (g, w)) in out.iter().zip(WANT_OUT.iter()).enumerate() {
            assert!((g - w).abs() < 2e-5, "hc_post[{i}]: got {g} want {w}");
        }
    }

    /// comb is summed over its FIRST index in hc_post. Transposing it is silent (both are
    /// hc x hc) and would corrupt the residual stream, so pin the asymmetry directly.
    #[test]
    fn hc_post_sums_comb_over_its_first_index() {
        let mut comb = [[0f32; MAX_HC]; MAX_HC];
        comb[0][1] = 1.0; // j=0 -> k=1 only
        let post = [0f32; MAX_HC];
        let residual = [1.0, 2.0, /* j=1 */ 30.0, 40.0];
        let mut out = vec![0f32; 4];
        hc_post(&[0.0, 0.0], &residual, &post, &comb, 2, 2, &mut out);
        // k=1 must pick up j=0's row (1,2), NOT j=1's (30,40).
        assert_eq!(&out[2..4], &[1.0, 2.0], "comb was transposed");
        assert_eq!(&out[0..2], &[0.0, 0.0]);
    }
}

