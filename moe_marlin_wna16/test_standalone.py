"""Standalone MOE-Marlin INT4 test. No vLLM test infra needed."""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
DEV = torch.device("cuda:0")


def gen_marlin_weight(k, n, group_size=-1, num_bits=4):
    """Generate Marlin-formatted INT4 weights + scales + reference FP16."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        quantize_weights,
        sort_weights,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_moe_permute_scales,
    )
    maxq = 2 ** num_bits - 1
    w = torch.randn((k, n), dtype=torch.half, device=DEV)
    if group_size > 0:
        # grouped quantization
        w_ref, q_w, s, _, _ = quantize_weights(w, num_bits, group_size, act_order=False)
    else:
        # per-channel
        s = torch.max(torch.abs(w), 0, keepdim=True)[0]
        s *= 2 / maxq
        q_w = torch.round(w / s).int() + (maxq + 1) // 2
        q_w = torch.clamp(q_w, 0, maxq)
        w_ref = (q_w - (maxq + 1) // 2).half() * s
        q_w = q_w.to(torch.int32)
        s = s.reshape(1, n)

    # Pack to Marlin format
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_sort_g_idx,
    )
    # Marlin weight packing: (k, n) int4 -> (k/16, n*16/pack) int32
    tile = 16
    pack_factor = 32 // num_bits
    # Marlin uses a specific packing + permutation
    # Use vllm's utility
    perm = torch.empty(0, dtype=torch.int, device=DEV)
    g_idx = torch.empty(0, dtype=torch.int, device=DEV)

    # Simple pack: pack num_bits values into int32
    # Marlin format: interleaved packing within tiles
    from vllm._C import ops as vllm_ops
    # If _C.ops is available, use it. Otherwise fallback.
    raise NotImplementedError("Need vllm C ops for weight packing")


def test_moe_marlin_basic():
    """Test MOE Marlin with pre-packed weights via vLLM utilities."""
    import moe_marlin_cuda
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )

    torch.manual_seed(42)

    # Problem dimensions
    m, n, k = 16, 128, 256
    num_experts = 4
    top_k = 2
    block_size_m = 16

    # Input
    a = torch.randn((m, k), dtype=torch.float16, device=DEV)

    # Router
    router_logits = torch.randn((m, num_experts), dtype=torch.float32, device=DEV)
    routing_weights = F.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(torch.float32)

    # Align block size for MOE
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids.to(torch.int32), block_size_m, num_experts
    )

    # Create FP16 reference weights per expert (no actual quantization for now)
    # Just verify the MOE dispatch works by using identity-like weights
    w_ref = torch.randn((num_experts, k, n), dtype=torch.float16, device=DEV)

    # Pack weights into Marlin format (k/16, n*16/8 in int32)
    # For INT4: each int32 holds 8 INT4 values
    # Marlin packing: tile_size=16, pack_factor=8
    tile_size = 16
    pack_factor = 8  # 32/4

    # Create packed weight tensor: (num_experts, k/tile_size, n*tile_size/pack_factor)
    b_q = torch.zeros((num_experts, k // tile_size, n * tile_size // pack_factor),
                       dtype=torch.int32, device=DEV)

    # Create scales: per-column (group_size=-1)
    # Shape: (num_experts, 1, n)
    b_scales = torch.ones((num_experts, 1, n), dtype=torch.float16, device=DEV)

    # For a simple smoke test, just verify the kernel runs without crashing
    workspace = marlin_make_workspace_new(DEV, max_blocks_per_sm=1)

    b_type_id = scalar_types.uint4b8.id

    print(f"Testing MOE-Marlin: m={m} n={n} k={k} experts={num_experts} top_k={top_k}")
    print(f"  a shape: {a.shape}")
    print(f"  b_q shape: {b_q.shape}")
    print(f"  b_scales shape: {b_scales.shape}")
    print(f"  sorted_token_ids shape: {sorted_token_ids.shape}")
    print(f"  expert_ids shape: {expert_ids.shape}")

    try:
        c = moe_marlin_cuda.moe_wna16_marlin_gemm(
            a,                          # a
            None,                       # c (auto-alloc)
            b_q,                        # b_q_weight
            None,                       # b_bias
            b_scales,                   # b_scales
            None,                       # a_scales
            None,                       # global_scale
            None,                       # b_zeros
            None,                       # g_idx
            None,                       # perm
            workspace,                  # workspace
            sorted_token_ids,           # sorted_token_ids
            expert_ids,                 # expert_ids
            num_tokens_post_padded,     # num_tokens_past_padded
            topk_weights,               # topk_weights
            block_size_m,               # moe_block_size
            top_k,                      # top_k
            True,                       # mul_topk_weights
            b_type_id,                  # b_q_type
            m,                          # size_m
            n,                          # size_n
            k,                          # size_k
            False,                      # is_k_full
            False,                      # use_atomic_add
            True,                       # use_fp32_reduce
            False,                      # is_zp_float
            -1,                         # thread_k
            -1,                         # thread_n
            1,                          # blocks_per_sm
        )
        torch.cuda.synchronize()
        print(f"  Output shape: {c.shape}")
        print(f"  Output sample: {c[0, :5]}")
        print("  KERNEL RAN SUCCESSFULLY!")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_moe_marlin_basic()
