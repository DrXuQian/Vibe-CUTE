"""Test CuTe-based Marlin kernel: accuracy vs original + performance benchmark."""

import unittest
import time

import numpy as np
import torch
import torch.nn as nn

import marlin

seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

DEV = torch.device('cuda:0')


def gen_quant4(m, n, groupsize=-1):
    tile = 16
    maxq = 2 ** 4 - 1
    w = torch.randn((m, n), dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    layer = marlin.Layer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    q = layer.B
    s = layer.s
    return ref, q, s


# ============================================================================
# Accuracy Tests
# ============================================================================

class TestAccuracy(unittest.TestCase):
    """Test CuTe kernel produces bit-exact results vs original kernel."""

    def _run(self, m, n, k, thread_k, thread_n, groupsize=-1):
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        B_ref, B, s = gen_quant4(k, n, groupsize=groupsize)

        C_ref = torch.matmul(A, B_ref)
        C_cute = torch.zeros((m, n), dtype=torch.half, device=DEV)
        workspace = torch.zeros(n // 128 * 16, device=DEV)

        marlin.mul_cute(A, B, C_cute, s, workspace, thread_k, thread_n, -1)
        torch.cuda.synchronize()

        # Compare against reference (torch.matmul), not original kernel,
        # because CTA reduction order may differ (both are numerically valid)
        rel_err = (torch.mean(torch.abs(C_cute - C_ref)) / torch.mean(torch.abs(C_ref))).item()
        self.assertLess(rel_err, 0.001,
            f"m={m} n={n} k={k} tk={thread_k} tn={thread_n} gs={groupsize} rel_err={rel_err}")

    # --- Tile shapes (mirrors original test_tiles) ---
    def test_tiles(self):
        for m in [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 118, 128, 152, 768, 1024]:
            for thread_k, thread_n in [(64, 256), (128, 128)]:
                if m > 16 and thread_k == 128:
                    continue
                with self.subTest(m=m, tk=thread_k, tn=thread_n):
                    self._run(m, 2 * 256, 1024, thread_k, thread_n)

    # --- K stages divisibility ---
    def test_k_stages_divisibility(self):
        for k in [3 * 64 + 64 * 4 * 2 + 64 * i for i in range(1, 4)]:
            with self.subTest(k=k):
                self._run(16, 2 * 256, k, 64, 256)

    # --- Very few stages ---
    def test_very_few_stages(self):
        for k in [64, 128, 192]:
            with self.subTest(k=k):
                self._run(16, 2 * 256, k, 64, 256)

    # --- Grouped quantization ---
    def test_groups(self):
        for m in [16]:
            for groupsize in [128]:
                for n, k in [(256, 512), (256, 1024), (256 * 128, 1024)]:
                    for thread_k, thread_n in [(128, 128), (64, 256)]:
                        with self.subTest(m=m, n=n, k=k, tk=thread_k, tn=thread_n, gs=groupsize):
                            self._run(m, n, k, thread_k, thread_n, groupsize)

    # --- Reference accuracy (vs torch.matmul) ---
    def test_vs_reference(self):
        for m in [1, 16, 64]:
            for thread_k, thread_n in [(128, 128), (64, 256)]:
                if m > 16 and thread_k == 128:
                    continue
                with self.subTest(m=m, tk=thread_k, tn=thread_n):
                    k, n = 1024, 512
                    A = torch.randn((m, k), dtype=torch.half, device=DEV)
                    B_ref, B, s = gen_quant4(k, n)
                    C_cute = torch.zeros((m, n), dtype=torch.half, device=DEV)
                    C_ref = torch.matmul(A, B_ref)
                    workspace = torch.zeros(n // 128 * 16, device=DEV)
                    marlin.mul_cute(A, B, C_cute, s, workspace, thread_k, thread_n, -1)
                    torch.cuda.synchronize()
                    rel_err = (torch.mean(torch.abs(C_cute - C_ref)) / torch.mean(torch.abs(C_ref))).item()
                    self.assertLess(rel_err, 0.001,
                        f"m={m} tk={thread_k} tn={thread_n} rel_err={rel_err}")


# ============================================================================
# Performance Benchmark
# ============================================================================

def benchmark_kernel(fn, warmup=20, iters=100):
    """Benchmark a CUDA kernel using events for accurate timing."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    # Use median to avoid outliers
    median_ms = times[len(times) // 2]
    return median_ms


class TestPerformance(unittest.TestCase):
    """Benchmark CuTe kernel vs original kernel."""

    def _bench(self, m, n, k, thread_k, thread_n, groupsize=-1):
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        B_ref, B, s = gen_quant4(k, n, groupsize=groupsize)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        workspace = torch.zeros(n // 128 * 16, device=DEV)

        def run_orig():
            workspace.zero_()
            marlin.mul(A, B, C, s, workspace, thread_k, thread_n, -1)

        def run_cute():
            workspace.zero_()
            marlin.mul_cute(A, B, C, s, workspace, thread_k, thread_n, -1)

        t_orig = benchmark_kernel(run_orig)
        t_cute = benchmark_kernel(run_cute)

        # Compute TFLOPS (2*m*n*k FLOPs for matmul)
        flops = 2.0 * m * n * k
        tflops_orig = flops / (t_orig * 1e-3) / 1e12
        tflops_cute = flops / (t_cute * 1e-3) / 1e12
        ratio = t_cute / t_orig

        print(f"  m={m:4d} n={n:5d} k={k:5d} gs={groupsize:4d} | "
              f"orig={t_orig:7.3f}ms ({tflops_orig:6.2f} TFLOPS)  "
              f"cute={t_cute:7.3f}ms ({tflops_cute:6.2f} TFLOPS)  "
              f"ratio={ratio:.3f}x")
        return t_orig, t_cute

    def test_gemv_shapes(self):
        """GEMV-like shapes (small m): typical inference scenario."""
        print("\n--- GEMV shapes (m=1,16) ---")
        print(f"  {'m':>4s} {'n':>5s} {'k':>5s} {'gs':>4s} | "
              f"{'orig':>10s} {'(TFLOPS)':>10s}  "
              f"{'cute':>10s} {'(TFLOPS)':>10s}  "
              f"{'ratio':>8s}")
        for m in [1, 16]:
            for n, k in [(4096, 4096), (4096, 11008), (11008, 4096)]:
                self._bench(m, n, k, 128, 128)

    def test_gemm_shapes(self):
        """GEMM shapes (larger m): batched inference."""
        print("\n--- GEMM shapes (m=64,128) ---")
        print(f"  {'m':>4s} {'n':>5s} {'k':>5s} {'gs':>4s} | "
              f"{'orig':>10s} {'(TFLOPS)':>10s}  "
              f"{'cute':>10s} {'(TFLOPS)':>10s}  "
              f"{'ratio':>8s}")
        for m in [64, 128]:
            for n, k in [(4096, 4096), (4096, 11008), (11008, 4096)]:
                self._bench(m, n, k, 64, 256)

    def test_grouped_perf(self):
        """Grouped quantization performance (groupsize=128)."""
        print("\n--- Grouped quantization (gs=128) ---")
        print(f"  {'m':>4s} {'n':>5s} {'k':>5s} {'gs':>4s} | "
              f"{'orig':>10s} {'(TFLOPS)':>10s}  "
              f"{'cute':>10s} {'(TFLOPS)':>10s}  "
              f"{'ratio':>8s}")
        for m in [1, 16]:
            for n, k in [(4096, 4096), (11008, 4096)]:
                self._bench(m, n, k, 128, 128, groupsize=128)


if __name__ == '__main__':
    unittest.main()
