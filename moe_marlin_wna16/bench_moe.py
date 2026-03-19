"""Benchmark MOE CuTe kernel vs original MOE kernel."""
import sys, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/root/marlin/marlin_w4a16')
sys.path.insert(0, '/root/marlin/moe_marlin_wna16')

DEV = torch.device("cuda:0")


def pack_marlin_weight(k, n, gs=-1):
    import importlib.util
    spec = importlib.util.spec_from_file_location('m', '/root/marlin/marlin_w4a16/__init__.py')
    mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
    maxq = 15
    w = torch.randn((k, n), dtype=torch.half, device=DEV)
    if gs != -1:
        w2 = w.reshape(-1, gs, n).permute(1, 0, 2).reshape(gs, -1)
        s = torch.max(torch.abs(w2), 0, keepdim=True)[0] * 2 / maxq
        w_int = torch.clamp(torch.round(w2 / s).int() + 8, 0, 15)
        ref = (w_int - 8).half() * s
        def rs(x): return x.reshape(gs, -1, n).permute(1, 0, 2).reshape(k, n).contiguous()
        ref = rs(ref); w_int = rs(w_int.half()).int()
        s = s.reshape(-1, n).contiguous()
    else:
        s = torch.max(torch.abs(w), 0, keepdim=True)[0] * 2 / maxq
        w_int = torch.clamp(torch.round(w / s).int() + 8, 0, 15)
        ref = (w_int - 8).half() * s
        s = s.reshape(-1, n).contiguous()
    linear = nn.Linear(k, n); linear.weight.data = ref.t()
    layer = mm.Layer(256, 256, groupsize=gs)
    if gs == -1: gs = k
    layer.k = k; layer.n = n; layer.groupsize = gs
    layer.B = torch.empty((k // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((k // gs, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    return ref, layer.B, layer.s


def benchmark_kernel(fn, warmup=20, iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start[i].record(); fn(); end[i].record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in zip(start, end)])
    return times[len(times) // 2]


def bench_moe(m, n, k, num_experts, top_k, gs):
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    import moe_marlin_cuda
    import moe_marlin_cute_cuda

    torch.manual_seed(42)
    a = torch.randn((m, k), dtype=torch.half, device=DEV)
    router = torch.randn((m, num_experts), dtype=torch.float32, device=DEV)
    rw = F.softmax(router, dim=-1)
    tw, ti = torch.topk(rw, top_k, dim=-1)
    tw = (tw / tw.sum(dim=-1, keepdim=True)).float()
    sorted_ids, expert_ids, ntp = moe_align_block_size(ti.int(), 16, num_experts)

    w_q_list, w_s_list = [], []
    for e in range(num_experts):
        _, q, s = pack_marlin_weight(k, n, gs=gs)
        w_q_list.append(q); w_s_list.append(s)
    w_q = torch.stack(w_q_list); w_s = torch.stack(w_s_list)
    workspace_orig = marlin_make_workspace_new(DEV, max_blocks_per_sm=4)
    workspace_cute = torch.zeros(max(n // 128 * 64, 1024), device=DEV, dtype=torch.int)
    b_type_id = scalar_types.uint4b8.id

    def run_orig():
        workspace_orig.zero_()
        return moe_marlin_cuda.moe_wna16_marlin_gemm(
            a, None, w_q, None, w_s, None, None, None, None, None,
            workspace_orig, sorted_ids, expert_ids, ntp, tw,
            16, top_k, True, b_type_id, m, n, k,
            True, False, True, False, -1, -1, 1)

    def run_cute():
        workspace_cute.zero_()
        return moe_marlin_cute_cuda.moe_marlin_cute_gemm(
            a, w_q, w_s, sorted_ids, expert_ids, ntp, tw, workspace_cute,
            top_k, True, m, n, k, gs)

    t_orig = benchmark_kernel(run_orig)
    t_cute = benchmark_kernel(run_cute)
    flops = 2.0 * m * n * k * top_k
    return t_orig, t_cute, flops


if __name__ == "__main__":
    # Build both modules
    import subprocess
    subprocess.run(["python", "setup.py", "build_ext", "--inplace"],
                   cwd="/root/marlin/marlin_w4a16",
                   env={**os.environ, "TORCH_CUDA_ARCH_LIST": "8.0;12.0"},
                   capture_output=True)

    print("MOE-Marlin Performance Benchmark")
    print(f"{'m':>4s} {'n':>5s} {'k':>5s} {'E':>2s} {'k':>2s} {'gs':>4s} | "
          f"{'orig':>10s} {'cute':>10s} {'ratio':>8s}")
    print("-" * 70)

    for m in [1, 16, 32]:
        for n, k in [(4096, 4096), (4096, 11008)]:
            for num_experts, top_k in [(8, 2)]:
                gs = 128
                try:
                    t_o, t_c, flops = bench_moe(m, n, k, num_experts, top_k, gs)
                    tflops_o = flops / (t_o * 1e-3) / 1e12
                    tflops_c = flops / (t_c * 1e-3) / 1e12
                    ratio = t_c / t_o
                    print(f"m={m:3d} n={n:5d} k={k:5d} E={num_experts} k={top_k} gs={gs:4d} | "
                          f"orig={t_o:7.3f}ms cute={t_c:7.3f}ms ratio={ratio:.3f}x")
                except Exception as e:
                    print(f"m={m:3d} n={n:5d} k={k:5d} E={num_experts} k={top_k} gs={gs:4d} | ERROR: {e}")
