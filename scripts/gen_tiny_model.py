#!/usr/bin/env python3
"""Generate a tiny GLM-shaped snapshot (config.json + model.safetensors) to
smoke-test `coli load`. Mirrors the dims in the Rust integration test."""
import json, struct, sys, os

D, NL, H, E = 8, 2, 2, 4
MOE_INTER, DENSE_INTER, FIRST_DENSE = 4, 8, 1
Q_LORA, KV_LORA, QK_NOPE, QK_ROPE, V_HEAD, N_SHARED, VOCAB = 4, 4, 4, 2, 4, 1, 10
QK_HEAD = QK_NOPE + QK_ROPE
S_I = MOE_INTER * N_SHARED

out = sys.argv[1]
os.makedirs(out, exist_ok=True)

cfg = {
    "hidden_size": D, "num_hidden_layers": NL, "num_attention_heads": H,
    "n_routed_experts": E, "num_experts_per_tok": 2, "moe_intermediate_size": MOE_INTER,
    "intermediate_size": DENSE_INTER, "first_k_dense_replace": FIRST_DENSE,
    "q_lora_rank": Q_LORA, "kv_lora_rank": KV_LORA, "qk_nope_head_dim": QK_NOPE,
    "qk_rope_head_dim": QK_ROPE, "v_head_dim": V_HEAD, "n_shared_experts": N_SHARED,
    "vocab_size": VOCAB, "n_group": 1, "topk_group": 1, "norm_topk_prob": False,
    "rms_norm_eps": 1e-5, "routed_scaling_factor": 1.0,
    "rope_parameters": {"rope_theta": 10000.0}, "eos_token_id": [9],
    "index_topk": 0, "index_n_heads": 0, "index_head_dim": 0,
}
json.dump(cfg, open(os.path.join(out, "config.json"), "w"))

tensors = [("model.embed_tokens.weight", VOCAB*D), ("lm_head.weight", VOCAB*D),
           ("model.norm.weight", D)]
for i in range(NL):
    p = lambda s: f"model.layers.{i}.{s}"
    tensors += [(p("input_layernorm.weight"), D), (p("post_attention_layernorm.weight"), D),
        (p("self_attn.q_a_proj.weight"), Q_LORA*D), (p("self_attn.q_a_layernorm.weight"), Q_LORA),
        (p("self_attn.q_b_proj.weight"), H*QK_HEAD*Q_LORA),
        (p("self_attn.kv_a_proj_with_mqa.weight"), (KV_LORA+QK_ROPE)*D),
        (p("self_attn.kv_a_layernorm.weight"), KV_LORA),
        (p("self_attn.kv_b_proj.weight"), H*(QK_NOPE+V_HEAD)*KV_LORA),
        (p("self_attn.o_proj.weight"), D*H*V_HEAD)]
    if i < FIRST_DENSE:
        tensors += [(p("mlp.gate_proj.weight"), DENSE_INTER*D), (p("mlp.up_proj.weight"), DENSE_INTER*D),
            (p("mlp.down_proj.weight"), D*DENSE_INTER)]
    else:
        tensors += [(p("mlp.gate.weight"), E*D), (p("mlp.gate.e_score_correction_bias"), E),
            (p("mlp.shared_experts.gate_proj.weight"), S_I*D), (p("mlp.shared_experts.up_proj.weight"), S_I*D),
            (p("mlp.shared_experts.down_proj.weight"), D*S_I)]

header, off, payload = {}, 0, bytearray()
for name, numel in tensors:
    nb = numel*4
    header[name] = {"dtype": "F32", "shape": [numel], "data_offsets": [off, off+nb]}
    off += nb
    for k in range(numel):
        payload += struct.pack("<f", ((k % 5) - 2) * 0.1)
hb = json.dumps(header).encode()
with open(os.path.join(out, "model.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(hb))); f.write(hb); f.write(payload)
print("wrote tiny model to", out, "-", len(tensors), "tensors")
