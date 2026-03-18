"""Test CuTe-based Marlin kernel vs original kernel."""

import torch
import torch.nn as nn
import marlin

seed = 0
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


def test_cute_vs_original(m, n, k, thread_k=-1, thread_n=-1, groupsize=-1):
    A = torch.randn((m, k), dtype=torch.half, device=DEV)
    B_ref, B, s = gen_quant4(k, n, groupsize=groupsize)

    C_orig = torch.zeros((m, n), dtype=torch.half, device=DEV)
    C_cute = torch.zeros((m, n), dtype=torch.half, device=DEV)
    workspace = torch.zeros(n // 128 * 16, device=DEV)

    # Original kernel
    marlin.mul(A, B, C_orig, s, workspace, thread_k, thread_n, -1)
    torch.cuda.synchronize()

    # Reset workspace
    workspace.zero_()

    # CuTe kernel
    marlin.mul_cute(A, B, C_cute, s, workspace, thread_k, thread_n, -1)
    torch.cuda.synchronize()

    # Compare
    max_diff = torch.max(torch.abs(C_orig - C_cute)).item()
    mean_diff = torch.mean(torch.abs(C_orig - C_cute)).item()
    rel_err = mean_diff / (torch.mean(torch.abs(C_orig)).item() + 1e-10)

    status = "PASS" if max_diff < 1e-3 else "FAIL"
    print(f"  m={m:4d} n={n:5d} k={k:5d} tk={thread_k:4d} tn={thread_n:4d} gs={groupsize:4d} | "
          f"max_diff={max_diff:.6f} rel_err={rel_err:.6f} [{status}]")
    return max_diff < 1e-3


if __name__ == '__main__':
    print("=== CuTe vs Original Marlin Kernel Test ===\n")

    all_pass = True

    # Test basic shapes with both tile configs
    print("--- Basic shapes ---")
    for m in [1, 16, 64, 128]:
        for thread_k, thread_n in [(128, 128), (64, 256)]:
            if m > 16 and thread_k == 128:
                continue
            ok = test_cute_vs_original(m, 512, 1024, thread_k, thread_n)
            all_pass = all_pass and ok

    # Test grouped quantization
    print("\n--- Grouped quantization (groupsize=128) ---")
    for m in [16]:
        for n, k in [(256, 512), (256, 1024)]:
            for thread_k, thread_n in [(128, 128), (64, 256)]:
                ok = test_cute_vs_original(m, n, k, thread_k, thread_n, groupsize=128)
                all_pass = all_pass and ok

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
