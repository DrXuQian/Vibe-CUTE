"""Standalone MOE-Marlin INT4 test using our W4A16 packing utility."""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/root/marlin/marlin_w4a16')
DEV = torch.device("cuda:0")


def pack_marlin_weight(k, n, groupsize=-1):
    """Pack random INT4 weights into Marlin format using W4A16 Layer.pack()."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('marlin_w4a16', '/root/marlin/marlin_w4a16/__init__.py')
    marlin_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(marlin_mod)

    maxq = 2 ** 4 - 1
    w = torch.randn((k, n), dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w_int = torch.round(w / s).int()
    w_int += (maxq + 1) // 2
    w_int = torch.clamp(w_int, 0, maxq)
    ref = (w_int - (maxq + 1) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n)).contiguous()
            return w
        ref = reshape(ref)
        w_int = reshape(w_int)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(k, n)
    linear.weight.data = ref.t()
    layer = marlin_mod.Layer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = k
    layer.k = k
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((k // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((k // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    return ref, layer.B, layer.s


def test_moe_marlin():
    """Test MOE Marlin INT4 kernel against FP16 reference."""
    sys.path.insert(0, '/root/marlin/moe_marlin_wna16')
    import moe_marlin_cuda
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )

    torch.manual_seed(42)

    m, n, k = 32, 256, 512
    num_experts = 4
    top_k = 2
    block_size_m = 16
    groupsize = 128

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

    # Create per-expert Marlin-packed weights
    w_ref_list, w_q_list, w_s_list = [], [], []
    for e in range(num_experts):
        ref, q, s = pack_marlin_weight(k, n, groupsize=groupsize)
        w_ref_list.append(ref)
        w_q_list.append(q)
        w_s_list.append(s)

    w_ref = torch.stack(w_ref_list)
    w_q = torch.stack(w_q_list)
    w_s = torch.stack(w_s_list)

    # FP16 reference computation
    c_ref = torch.zeros((m, n), dtype=torch.float16, device=DEV)
    for i in range(m):
        for t in range(top_k):
            eid = topk_ids[i, t].item()
            wt = topk_weights[i, t].item()
            c_ref[i] += wt * (a[i:i+1] @ w_ref[eid]).squeeze(0)

    # Workspace
    workspace = marlin_make_workspace_new(DEV, max_blocks_per_sm=1)
    b_type_id = scalar_types.uint4b8.id

    print(f"MOE-Marlin test: m={m} n={n} k={k} experts={num_experts} top_k={top_k} gs={groupsize}")

    try:
        c_marlin = moe_marlin_cuda.moe_wna16_marlin_gemm(
            a, None, w_q, None, w_s,
            None, None, None, None, None,
            workspace, sorted_token_ids, expert_ids,
            num_tokens_post_padded, topk_weights,
            block_size_m, top_k, True,
            b_type_id, m, n, k,
            True, False, True, False,
            -1, -1, 1,
        )
        torch.cuda.synchronize()

        # c_marlin shape: (m*top_k, n) — topk_weights already applied per row
        # Need to sum over top_k for each token
        c_marlin_2d = c_marlin.view(m, top_k, n)
        c_marlin_reduced = c_marlin_2d.sum(dim=1)

        rel_err = (torch.abs(c_marlin_reduced - c_ref).mean() / torch.abs(c_ref).mean()).item()
        max_diff = torch.abs(c_marlin_reduced - c_ref).max().item()
        print(f"  rel_err={rel_err:.6f} max_diff={max_diff:.4f}")
        print(f"  c_ref[:2,:4]    = {c_ref[:2,:4]}")
        print(f"  c_marlin[:2,:4] = {c_marlin_reduced[:2,:4]}")
        if rel_err < 0.01:
            print("  PASSED!")
            return True
        else:
            print(f"  FAILED")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return False


if __name__ == "__main__":
    # First rebuild the module
    import subprocess
    subprocess.run(["python", "setup.py", "build_ext", "--inplace"],
                   cwd="/root/marlin/marlin_w4a16",
                   env={**__import__('os').environ, "TORCH_CUDA_ARCH_LIST": "8.0;12.0"},
                   capture_output=True)
    ok = test_moe_marlin()
    exit(0 if ok else 1)
