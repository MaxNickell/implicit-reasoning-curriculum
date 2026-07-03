import torch
from flash_attn import flash_attn_varlen_func

from ..model.arch import ModelParts


_apply_rope = None
for _rope_module in [
    'transformers.models.qwen3.modeling_qwen3',
    'transformers.models.gpt_neox.modeling_gpt_neox',
]:
    try:
        _mod = __import__(_rope_module, fromlist=['apply_rotary_pos_emb'])
        _apply_rope = _mod.apply_rotary_pos_emb
        break
    except (ImportError, AttributeError):
        continue


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def forward_layer_varlen(
    layer,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    parts: ModelParts,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    seq_len = hidden_states.shape[0]
    num_heads = parts.num_heads
    num_kv_heads = parts.num_kv_heads
    head_dim = parts.head_dim
    arch = parts.arch
    parallel_residual = parts.parallel_residual
    rotary_ndims = parts.rotary_ndims

    residual = hidden_states
    ln1_out = layer.input_layernorm(hidden_states)

    # QKV projection
    if arch == 'gpt_neox':
        # GPT-NeoX uses interleaved QKV: [q1|k1|v1 | q2|k2|v2 | ...] per head
        qkv = layer.attention.query_key_value(ln1_out)  # [seq, 3 * H]
        qkv = qkv.view(seq_len, num_heads, 3 * head_dim)
        q = qkv[..., :head_dim]              # [seq, num_heads, head_dim]
        k = qkv[..., head_dim:2*head_dim]
        v = qkv[..., 2*head_dim:]
    else:
        q = layer.self_attn.q_proj(ln1_out)
        k = layer.self_attn.k_proj(ln1_out)
        v = layer.self_attn.v_proj(ln1_out)
        # Reshape: [seq_len, num_heads, head_dim]
        q = q.view(seq_len, num_heads, head_dim)
        k = k.view(seq_len, num_kv_heads, head_dim)
        v = v.view(seq_len, num_kv_heads, head_dim)

    # QK Normalization (Qwen3 specific)
    if arch == 'qwen':
        if hasattr(layer.self_attn, 'q_norm') and layer.self_attn.q_norm is not None:
            q = layer.self_attn.q_norm(q)
            k = layer.self_attn.k_norm(k)

    # Add batch dim and transpose for RoPE: [1, num_heads, seq_len, head_dim]
    q = q.unsqueeze(0).transpose(1, 2)
    k = k.unsqueeze(0).transpose(1, 2)

    # Handle partial RoPE (e.g. Pythia rotary_pct=0.25)
    partial_rope = rotary_ndims is not None and rotary_ndims < head_dim
    if partial_rope:
        q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
        k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]
    else:
        q_rot, k_rot = q, k

    # Apply pre-computed RoPE cos/sin
    if _apply_rope is not None:
        q_rot, k_rot = _apply_rope(q_rot, k_rot, cos, sin)
    else:
        cos_u = cos.unsqueeze(1)
        sin_u = sin.unsqueeze(1)
        q_rot = (q_rot * cos_u) + (_rotate_half(q_rot) * sin_u)
        k_rot = (k_rot * cos_u) + (_rotate_half(k_rot) * sin_u)

    if partial_rope:
        q = torch.cat((q_rot, q_pass), dim=-1)
        k = torch.cat((k_rot, k_pass), dim=-1)
    else:
        q, k = q_rot, k_rot

    # Reshape for flash_attn_varlen: [seq_len, num_heads, head_dim]
    q = q.squeeze(0).transpose(0, 1).contiguous()
    k = k.squeeze(0).transpose(0, 1).contiguous()
    v = v.contiguous()

    # Flash attention with per-sequence boundaries
    attn_output = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )

    # Output projection
    attn_output = attn_output.reshape(seq_len, num_heads * head_dim)
    if arch == 'gpt_neox':
        attn_output = layer.attention.dense(attn_output)
    else:
        attn_output = layer.self_attn.o_proj(attn_output)

    # Residual + MLP
    if parallel_residual:
        # GPT-NeoX (Pythia): x = x + attn(ln1(x)) + mlp(ln2(x))
        ln2_out = layer.post_attention_layernorm(residual)
        mlp_output = layer.mlp(ln2_out)
        hidden_states = residual + attn_output + mlp_output
    else:
        # Qwen: x = x + attn(ln1(x)); x = x + mlp(ln2(x))
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    return hidden_states
