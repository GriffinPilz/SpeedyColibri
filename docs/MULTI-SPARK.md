# Multi-Spark (expert-parallel)

<!-- Extracted from README.md 2026-08-06 to keep the README to the six things a
reader needs first: speeds, models, how to run them, where they come from, what was
changed, and quality. Nothing here was trimmed in the move. -->

## Multi-Spark (expert-parallel)

**Working, and it's the single biggest win available.** The 256 experts/layer are
split across nodes: each Spark owns half, loads and computes only its own half, and
answers its peers over the ConnectX/RoCE fabric. The dense part (attention, shared
expert, embeddings) is replicated per node, so only expert activations cross the wire
(~24 KB each way, not expert weights).

**All four model families are supported**, including Nemotron-3-Super's hybrid
latent-MoE. Nemotron splits over its `moe_latent` space rather than `hidden` — its
routed experts consume the post-`fc1` latent vector — so it is *cheaper* to shard than
the others: fewer floats per token on the wire, with `fc2` and the shared expert staying
local as replicated dense weights.

**Measured EP overhead** (2-rank TCP loopback on one box, residency held constant so the
only variable is the exchange, 3 reps): **0.9% on prefill, ~15% on decode**,
token-identical to single-node. Prefill amortizes the per-layer round trips across the
whole batch; decode pays them for a single token, which is the entire difference. That
15% is measured with *both ranks sharing one GPU* — on two real Sparks the peer's expert
compute runs on its own GPU in parallel — so treat it as a pessimistic bound on the
transport cost, and small either way against the residency win below.

> ⚠️ **These 2-Spark numbers are superseded and not comparable to the record above.**
> They were taken with a *repeated single prompt* (which reads ~1.5–2× high because the
> cache hits on the repeat) on the pre-8/4 model. They are kept only because they are
> the sole 2-node data that exists; the shape (2-node ≈ 2× 1-node, from residency not
> compute) is believed to hold, but the magnitudes must be re-measured with diverse
> prompts on 8/4 (RDMA-A). Treat as illustrative, not current.

Measured on two DGX Sparks, 32-token greedy decode, 6 consecutive repeats against a
warm server, **40 GB expert cache per node**:

| | cold (1st run) | warm (converged) |
|---|---|---|
| 1 Spark | 0.71 tok/s | **0.76 tok/s** |
| 2 Sparks | 1.15 tok/s | **~1.95 tok/s** (**2.6×**) |

The cold/warm gap is the point: the first request pays for filling the cache, and a
`serve` warm-up prompt buys that back before real traffic arrives.

Output is **bit-identical** to single-node (all 32 tokens), verified on the real 744 B
model. The win is *residency, not compute*: at the same per-node budget each Spark
caches a 128-expert shard instead of all 256, so it hits disk far less. Fabric latency
is a rounding error by comparison — RoCE RTT is ~0.36 ms, so all 75 layers of
round-trips cost ~27 ms of a ~510 ms token (~5%).

### Running it

Start the **workers first** — the driver verifies every peer at startup and exits if
one is unreachable.

```bash
# --- on each worker node (rank 1..N-1) ---
COLI_NUM_NODES=2 COLI_NODE_RANK=1 \
  docker/run-dgx.sh worker                    # serves its shard on :48800

# --- on the driver (rank 0) — this is the node you send requests to ---
COLI_NUM_NODES=2 COLI_NODE_RANK=0 \
  COLI_PEERS=1=192.168.100.10:48800 \
  docker/run-dgx.sh serve 8080
```

`docker/run-dgx.sh cluster` scans the fabric and prints the Sparks it can see, with
their RoCE addresses — use it to fill in `COLI_PEERS`.

Both nodes print a **sharding fingerprint** at startup; they must match. They also
print their build revision (`coli v0.1.0 (abc1234)`) — that must match too, or one
node is running stale code. Nodes that disagree about the expert map are refused at
connect time rather than silently producing wrong tokens, so a mismatch is a startup
failure, never a wrong answer.

| Var | Meaning | Default |
|---|---|---|
| `COLI_NUM_NODES` | cluster size; `1` disables expert-parallel entirely | `1` |
| `COLI_NODE_RANK` | this node's rank, `0..NUM_NODES-1`. Rank 0 is the driver (runs `serve`); the rest run `worker` | `0` |
| `COLI_PEERS` | `rank=host:port` for **every** other rank, comma-separated. Required on the driver; a missing rank is a startup error | none |
| `COLI_EXPERT_PORT` | port a `worker` listens on | `48800` |
| `COLI_SHARD` | `hot` → assign experts to balance *traffic* rather than count, from the usage history. **Measured no gain on 2 nodes** (~1.96 vs ~1.95 tok/s) and it requires every node to share a byte-identical `.coli_usage` or the handshake refuses. Leave unset. | contiguous |
| `COLI_USAGE` | path to the usage history. Point every node at the *same* file (shared storage) if using `COLI_SHARD=hot` | `<snap>/.coli_usage` |

**Scaling past 2.** Per-node cache (~5 900 experts at 106 GB) versus 19 200 total
routed experts means at **4+ Sparks each node's whole shard is resident** and expert
streaming stops entirely. Sharding does *not* reduce disk footprint — every node
holds the full 403 GB snapshot and simply reads less of it.

**Next:** the RDMA transport (`colibri-cluster`, stubbed behind the same `Transport`
trait) would cut the ~27 ms/token of fabric latency — worth ~5%. The larger remaining
lever is still expert residency, i.e. more nodes.

