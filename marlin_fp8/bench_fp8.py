"""Benchmark FP8 Marlin kernel vs FP16 cuBLAS across various shapes."""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
DEV = torch.device("cuda:0")


def benchmark_kernel(fn, warmup=50, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in zip(start, end)])
    return times[len(times) // 2]  # median


def bench_fp8_marlin(m, n, k, group_size=128):
    """Benchmark FP8 Marlin kernel."""
    import marlin_fp8_cuda
    from vllm.scalar_type import scalar_types

    a = torch.randn((m, k), dtype=torch.half, device=DEV)

    # Create FP8 E4M3 weights (random, properly scaled)
    w_fp8 = torch.randn((k, n), dtype=torch.half, device=DEV)

    # Quantize to FP8 range and pack for Marlin
    # For benchmarking, we just need valid packed data — use random int32
    b_q = torch.randint(-2**31, 2**31 - 1, (k // 16, n * 16 // 4),
                         dtype=torch.int32, device=DEV)

    # Scales: (k/group_size, n) or (1, n) for per-channel
    num_groups = k // group_size if group_size > 0 else 1
    scales = torch.ones((num_groups, n), dtype=torch.half, device=DEV) * 0.1

    # Workspace
    workspace = torch.zeros(n // 64 * 16, device=DEV, dtype=torch.int)

    b_type = scalar_types.float8_e4m3fn

    def run():
        workspace.zero_()
        return marlin_fp8_cuda.marlin_gemm(
            a,              # a
            None,           # c_or_none
            b_q,            # b_q_weight
            None,           # b_bias
            scales,         # b_scales
            None,           # a_scales
            None,           # global_scale
            None,           # b_zeros
            None,           # g_idx
            None,           # perm
            workspace,      # workspace
            b_type.id,      # b_q_type
            m, n, k,
            True,           # is_k_full
            False,          # has_zp
            False,          # use_fp32_reduce
            False,          # is_zp_float
        )

    return benchmark_kernel(run)


def bench_cublas_fp16(m, n, k):
    """Benchmark cuBLAS FP16 GEMM (torch.matmul)."""
    a = torch.randn((m, k), dtype=torch.half, device=DEV)
    b = torch.randn((k, n), dtype=torch.half, device=DEV)

    def run():
        return torch.matmul(a, b)

    return benchmark_kernel(run)


if __name__ == "__main__":
    print("FP8 Marlin vs FP16 cuBLAS Benchmark")
    print(f"{'m':>5s} {'n':>6s} {'k':>6s} | {'Marlin FP8':>12s} {'cuBLAS FP16':>12s} {'speedup':>8s} | {'FP8 TFLOPS':>11s} {'FP16 TFLOPS':>12s}")
    print("-" * 85)

    shapes = [
        # GEMV shapes (m=1)
        (1, 4096, 4096),
        (1, 4096, 11008),
        (1, 11008, 4096),
        (1, 4096, 14336),
        (1, 14336, 4096),
        # Small batch
        (4, 4096, 4096),
        (4, 4096, 14336),
        (4, 14336, 4096),
        # Medium batch
        (16, 4096, 4096),
        (16, 4096, 14336),
        (16, 14336, 4096),
        # Larger batch
        (64, 4096, 4096),
        (64, 4096, 14336),
        (128, 4096, 4096),
        (128, 4096, 14336),
    ]

    for m, n, k in shapes:
        try:
            t_fp8 = bench_fp8_marlin(m, n, k)
        except Exception as e:
            t_fp8 = float('inf')
            print(f"m={m:4d} n={n:5d} k={k:5d} | FP8 ERROR: {e}")
            continue

        t_fp16 = bench_cublas_fp16(m, n, k)

        flops = 2.0 * m * n * k
        tflops_fp8 = flops / (t_fp8 * 1e-3) / 1e12
        tflops_fp16 = flops / (t_fp16 * 1e-3) / 1e12
        speedup = t_fp16 / t_fp8

        print(f"m={m:4d} n={n:5d} k={k:5d} | "
              f"{t_fp8:9.3f} ms {t_fp16:9.3f} ms {speedup:7.2f}x | "
              f"{tflops_fp8:9.2f} T  {tflops_fp16:10.2f} T")
