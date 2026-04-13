import math
import torch
import torch.nn.functional as F


def compute_attention_scores(query_states, key_states, pooling="max"):
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    query_group_size = q_heads // kv_heads

    # MHA
    if query_group_size == 1:
        return torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)

    # [batch_size, kv_heads, query_group_size, q_len, head_dim]
    query_states = query_states.view(
        batch_size, kv_heads, query_group_size, q_len, head_dim
    )

    # [batch_size, kv_heads, 1, kv_len, head_dim]
    key_states = key_states.unsqueeze(2)

    # [batch_size, kv_heads, query_group_size, q_len, kv_len]
    attn_weights = torch.matmul(
        query_states, key_states.transpose(3, 4)
    ) / math.sqrt(head_dim)

    # apply pooling over query_group_size dimension
    if pooling == "mean":
        attn_weights = attn_weights.mean(dim=2)
    elif pooling == "max":
        attn_weights = attn_weights.max(dim=2).values
    else:
        raise ValueError("Pooling method not supported")

    return attn_weights


def cal_similarity(
    key_states,
    threshold=0.5,
    retain_direction="last",
    chunk=256,
):
    device = key_states.device
    dtype = key_states.dtype

    k = F.normalize(key_states[0], dim=-1)  # [H, L, D]
    H, L, D = k.shape

    sum_sim = torch.zeros((H, L), device=device, dtype=dtype)

    for start in range(0, L, chunk):
        end = min(start + chunk, L)
        c = end - start

        k_chunk = k[:, start:end]  # [H, c, D]
        sim = torch.matmul(k_chunk, k.transpose(-1, -2))  # [H, c, L]

        sim[:, torch.arange(c), torch.arange(start, end, device=device)] = 0.0

        similarity_mask = sim > threshold
        if similarity_mask.any():
            indices = torch.where(
                similarity_mask,
                torch.arange(L, device=device),
                torch.zeros_like(similarity_mask, dtype=torch.long),
            )

            if retain_direction == "last":
                similarity_retain = torch.max(indices, dim=-1)[0]

            elif retain_direction == "first":
                similarity_retain = torch.min(indices, dim=-1)[0]

            else:
                raise ValueError("Invalid retain_direction")

            head_idx = torch.arange(H, device=device).unsqueeze(1).expand(H, c)
            row_idx = torch.arange(c, device=device).unsqueeze(0).expand(H, c)

            sim[head_idx, row_idx, similarity_retain] = 0.0

        sum_sim[:, start:end] = sim.sum(dim=-1)

    mean_sim = sum_sim / L
    return mean_sim.softmax(dim=-1)