"""Test CuTe MOE kernel vs original MOE kernel."""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/root/marlin/marlin_w4a16')
sys.path.insert(0, '/root/marlin/moe_marlin_wna16')

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
        def reshape(ww):
            ww = ww.reshape((groupsize, -1, n))
            ww = ww.permute(1, 0, 2)
            ww = ww.reshape((k, n)).contiguous()
            return ww
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


def test_cute_moe_vs_reference():
    """Test CuTe MOE kernel correctness against FP16 reference."""
    import moe_marlin_cute_cuda
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    torch.manual_seed(42)

    m, n, k = 32, 256, 512
    num_experts = 4
    top_k = 2
    block_size_m = 16
    groupsize = 128

    a = torch.randn((m, k), dtype=torch.float16, device=DEV)

    # Router
    router_logits = torch.randn((m, num_experts), dtype=torch.float32, device=DEV)
    routing_weights = F.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(torch.float32)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids.to(torch.int32), block_size_m, num_experts
    )

    # Per-expert packed weights
    w_ref_list, w_q_list, w_s_list = [], [], []
    for e in range(num_experts):
        ref, q, s = pack_marlin_weight(k, n, groupsize=groupsize)
        w_ref_list.append(ref)
        w_q_list.append(q)
        w_s_list.append(s)

    w_ref = torch.stack(w_ref_list)
    w_q = torch.stack(w_q_list)
    w_s = torch.stack(w_s_list)

    # FP16 reference
    c_ref = torch.zeros((m, n), dtype=torch.float16, device=DEV)
    for i in range(m):
        for t in range(top_k):
            eid = topk_ids[i, t].item()
            wt = topk_weights[i, t].item()
            c_ref[i] += wt * (a[i:i+1] @ w_ref[eid]).squeeze(0)

    # Workspace
    workspace = torch.zeros(n // 128 * 16, device=DEV, dtype=torch.int)

    print(f"CuTe MOE test: m={m} n={n} k={k} experts={num_experts} top_k={top_k} gs={groupsize}")

    try:
        c_cute = moe_marlin_cute_cuda.moe_marlin_cute_gemm(
            a, w_q, w_s,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            topk_weights, workspace,
            top_k, True,  # mul_topk_weights
            m, n, k, groupsize,
        )
        torch.cuda.synchronize()

        # c_cute: (m*top_k, n), sum over top_k
        c_cute_2d = c_cute.view(m, top_k, n)
        c_cute_reduced = c_cute_2d.sum(dim=1)

        rel_err = (torch.abs(c_cute_reduced - c_ref).mean() / torch.abs(c_ref).mean()).item()
        max_diff = torch.abs(c_cute_reduced - c_ref).max().item()
        print(f"  rel_err={rel_err:.6f} max_diff={max_diff:.4f}")
        print(f"  c_ref[:2,:4]  = {c_ref[:2,:4]}")
        print(f"  c_cute[:2,:4] = {c_cute_reduced[:2,:4]}")
        if rel_err < 0.01:
            print("  PASSED!")
            return True
        else:
            print("  FAILED")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return False


if __name__ == "__main__":
    # Rebuild W4A16 module (needed for pack utility)
    import subprocess, os
    subprocess.run(["python", "setup.py", "build_ext", "--inplace"],
                   cwd="/root/marlin/marlin_w4a16",
                   env={**os.environ, "TORCH_CUDA_ARCH_LIST": "8.0;12.0"},
                   capture_output=True)
    ok = test_cute_moe_vs_reference()
    exit(0 if ok else 1)
