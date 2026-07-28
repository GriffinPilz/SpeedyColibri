"""Independent port of fla `naive_recurrent_kda` (fla/ops/kda/naive.py), used to check
colibri's hardcoded expectations in kda.rs. Transcribed from the fla source:

    S = S * g_i[..., None].exp()
    S = S + einsum('b h k, b h v -> b h k v', b_i[...,None]*k_i, v_i - (k_i[...,None]*S).sum(-2))
    o[:, i] = einsum('b h k, b h k v -> b h v', q_i, S)

with q pre-scaled by K**-0.5 (the caller does that), and beta passed through sigmoid to
match colibri's `kda_recurrent`, which applies the sigmoid internally.
"""
import math

H, DK, S_LEN = 2, 3, 4
sig = lambda x: 1.0 / (1.0 + math.exp(-x))

qv = [((i * 7 % 13) / 13.0) - 0.5 for i in range(S_LEN * H * DK)]
kv = [((i * 5 % 11) / 11.0) - 0.5 for i in range(S_LEN * H * DK)]
vv = [((i * 3 % 17) / 17.0) - 0.5 for i in range(S_LEN * H * DK)]
gv = [-0.1 - 0.05 * (i * 2 % 5) for i in range(S_LEN * H * DK)]
bv = [0.3 * ((i % 4) - 1.5) for i in range(S_LEN * H)]

scale = DK ** -0.5
q = [a * scale for a in qv]

state = [[0.0] * (DK * DK) for _ in range(H)]   # per head, [K][V] flattened
o = [0.0] * (S_LEN * H * DK)

for t in range(S_LEN):
    row = t * H * DK
    for h in range(H):
        b0 = h * DK
        qh = q[row + b0: row + b0 + DK]
        kh = kv[row + b0: row + b0 + DK]
        vh = vv[row + b0: row + b0 + DK]
        gh = gv[row + b0: row + b0 + DK]
        Sh = state[h]
        # S = S * exp(g)   (g is per key-dim, broadcast across v)
        for kk in range(DK):
            e = math.exp(gh[kk])
            for x in range(DK):
                Sh[kk * DK + x] *= e
        # kts = (k[...,None] * S).sum(-2)  == k^T S
        kts = [sum(kh[kk] * Sh[kk * DK + x] for kk in range(DK)) for x in range(DK)]
        # S += (beta*k) outer (v - k^T S)
        b = sig(bv[t * H + h])
        for kk in range(DK):
            bk = b * kh[kk]
            for x in range(DK):
                Sh[kk * DK + x] += bk * (vh[x] - kts[x])
        # o = q^T S  (UPDATED S)
        for x in range(DK):
            o[row + b0 + x] = sum(qh[kk] * Sh[kk * DK + x] for kk in range(DK))

want_o = [
    -0.008449558, -0.005467361, -0.002485164, -0.001332029, -0.009324205, -0.01731638,
     0.02029327,   0.01201599,   0.003738716, -0.01205965,  -0.03982272,  -0.06758579,
    -0.003106576, -0.003155352, -0.003204128, -0.0005213997, 0.008427472,  0.08134121,
     0.01579968,   0.005727862, -0.004343956, -0.01423591,  -0.02219233,   0.06456562,
]
want_state = [0.08012541, 0.04224312, 0.004360832]

bad = 0
for i, (g, w) in enumerate(zip(o, want_o)):
    if abs(g - w) > 2e-6:
        bad += 1
        print(f"  o[{i}]: ref {g:.9g}  vs hardcoded {w:.9g}   diff {abs(g-w):.3g}")
for i, w in enumerate(want_state):
    g = state[0][i]
    if abs(g - w) > 2e-6:
        bad += 1
        print(f"  state[0,0,{i}]: ref {g:.9g}  vs hardcoded {w:.9g}")
print(f"\n{'MISMATCHES: %d' % bad if bad else 'ALL %d values match the fla reference' % (len(want_o)+len(want_state))}")
