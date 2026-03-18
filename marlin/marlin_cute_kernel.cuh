/*
 * Marlin FP16xINT4 GEMM kernel rewritten with CuTe.
 * Based on the original Marlin kernel by Elias Frantar.
 *
 * This implementation uses CuTe for:
 *   - Tensor layout definitions (global, shared, register)
 *   - GMEM->SMEM copies via TiledCopy + cp.async atoms
 *   - SMEM->Reg copies via ldmatrix atoms (A matrix)
 *   - MMA atom definition (SM80_16x8x16)
 *   - Swizzled shared memory layouts
 *
 * Manual implementations are kept for:
 *   - INT4 dequantization (lop3 trick)
 *   - CTA striped dispatcher
 *   - Thread-block and global reductions
 *   - Dequant-MMA interleaved compute loop
 */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

// CuTe headers
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/swizzle.hpp>

using namespace cute;

// ============================================================================
// Type Aliases - reuse Vec/FragA/FragB/FragC/FragS/I4 from marlin_cuda_kernel.cu
// ============================================================================

constexpr int ceildiv_cute(int a, int b) {
  return (a + b - 1) / b;
}

// ============================================================================
// PTX Intrinsics (unchanged from original Marlin)
// ============================================================================

// Predicated cp.async: global -> shared (cache at all levels)
__device__ inline void cp_async4_pred_cute(void* smem_ptr, const void* glob_ptr, bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .pred p;\n"
    "   setp.ne.b32 p, %0, 0;\n"
    "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
    "}\n" :: "r"((int) pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

// cp.async with cache-global hint (for weights B, accessed only once)
__device__ inline void cp_async4_stream_cute(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   cp.async.cg.shared.global [%0], [%1], %2;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

__device__ inline void cp_async_fence_cute() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait_cute() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// m16n8k16 MMA instruction
__device__ inline void mma_cute(const FragA& a_frag, const FragB& frag_b, FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b[0]),  "r"(b[1]),
       "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
  );
}

// ldmatrix: load 4x m8n8 matrix fragments from shared memory
__device__ inline void ldsm4_cute(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
  );
}

// lop3: 3-input logical operation
template <int lut>
__device__ inline int lop3_cute(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}

// INT4 dequantization: int32 -> 4x FP16 (FragB)
__device__ inline FragB dequant_cute(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  int lo = lop3_cute<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3_cute<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  FragB frag_b;
  frag_b[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag_b[1] = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );
  return frag_b;
}

// Scale a B fragment by quantization scale
__device__ inline void scale_cute(FragB& frag_b, FragS& frag_s, int i) {
  half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

// Cross-CTA synchronization barriers
__device__ inline void barrier_acquire_cute(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

__device__ inline void barrier_release_cute(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    asm volatile ("fence.acq_rel.gpu;\n");
    asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
  }
}

// ============================================================================
// Main CuTe-based Marlin Kernel
// ============================================================================

template <
  const int threads,           // threads per CTA (256)
  const int thread_m_blocks,   // M blocks per CTA (1-4, each 16 rows)
  const int thread_n_blocks,   // N blocks per CTA (8 or 16)
  const int thread_k_blocks,   // K blocks per CTA (4 or 8)
  const int stages,            // pipeline stages (4)
  const int group_blocks = -1  // quantization group size in blocks (-1 = per-column)
>
__global__ void MarlinCute(
  const int4* __restrict__ A,   // FP16 input: (M, K), row-major
  const int4* __restrict__ B,   // INT4 quantized weights: (K/16, N*16/32), special layout
        int4* __restrict__ C,   // FP16 output: (M, N), row-major
  const int4* __restrict__ s,   // FP16 scales: (K/groupsize, N/8), row-major
  int prob_m,
  int prob_n,
  int prob_k,
  int* locks
) {
  // ========================================================================
  // CTA Dispatcher
  // Uses immutable base pointers + par_m_idx offset instead of mutating
  // A/C/locks pointers directly.
  // ========================================================================
  int parallel = 1;
  if (prob_m > 16 * thread_m_blocks) {
    parallel = prob_m / (16 * thread_m_blocks);
    prob_m = 16 * thread_m_blocks;
  }

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters = ceildiv_cute(k_tiles * n_tiles * parallel, gridDim.x);
  if (group_blocks != -1)
    iters = (group_blocks / thread_k_blocks) * ceildiv_cute(iters, (group_blocks / thread_k_blocks));

  int slice_row = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;
  int slice_col = slice_col_par;
  int slice_iters;
  int slice_count = 0;
  int slice_idx;

  // Track M-parallel offset via index instead of pointer mutation
  // par_m_idx counts how many (16 * thread_m_blocks) M-blocks we've advanced
  int par_m_idx = 0;
  if (slice_col_par >= n_tiles) {
    par_m_idx = slice_col_par / n_tiles;
    slice_col = slice_col_par % n_tiles;
  }

  // Compute current A/C/locks from immutable base + par_m_idx
  auto cur_A = [&]() -> const int4* { return A + par_m_idx * 16 * thread_m_blocks * prob_k / 8; };
  auto cur_C = [&]() -> int4* { return C + par_m_idx * 16 * thread_m_blocks * prob_n / 8; };
  auto cur_locks = [&]() -> int* { return locks + par_m_idx * n_tiles; };

  auto init_slice = [&] () {
    slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
      slice_iters = 0;
    if (slice_iters == 0)
      return;
    if (slice_row + slice_iters > k_tiles)
      slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * ceildiv_cute(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv_cute(k_tiles - col_off, iters);
      if (col_off > 0)
        slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0)
          slice_idx--;
      }
    }
    if (slice_col == n_tiles) {
      par_m_idx++;  // advance to next M-parallel block
      slice_col = 0;
    }
  };
  init_slice();

  // ========================================================================
  // CuTe Layout Definitions
  // ========================================================================

  // --- A matrix parameters ---
  int a_gl_stride = prob_k / 8;  // A row stride in int4 (= prob_k / 8)
  constexpr int a_sh_stride  = 16 * thread_k_blocks / 8;  // smem row stride in int4
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);  // smem int4 per stage

  // --- B matrix parameters ---
  int b_gl_stride = 16 * prob_n / 32;  // B row stride in int4
  constexpr int b_sh_stride  = 32 * thread_n_blocks / 4;  // smem N-dimension in int4
  constexpr int b_sh_rd_delta = threads;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / threads;

  // --- Scale matrix parameters ---
  int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
  constexpr int s_sh_stage  = s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  // ========================================================================
  // CuTe Layouts and TiledCopy for A Matrix
  // ========================================================================

  // A tile dimensions (one pipeline stage)
  constexpr int A_TILE_M = 16 * thread_m_blocks;         // rows in half_t
  constexpr int A_TILE_K_HALF = 16 * thread_k_blocks;    // cols in half_t
  constexpr int A_TILE_K_INT4 = a_sh_stride;             // cols in int4 (= A_TILE_K_HALF / 8)
  constexpr int A_M_THREADS = threads / A_TILE_K_INT4;   // threads along M
  constexpr int A_SH_BITS = (A_TILE_K_INT4 == 16) ? 4 : 3;  // log2(int4 stride)

  // --- A Shared Memory Layout (half_t, with Swizzle) ---
  // Swizzle<B, 3, B> on half_t indices = Swizzle<B, 0, B> on int4 indices
  // (3 low bits select within int4, swizzle operates on int4-granularity bits)
  using SmemLayoutA = decltype(composition(
    Swizzle<A_SH_BITS, 3, A_SH_BITS>{},
    make_layout(make_shape(Int<A_TILE_M>{}, Int<A_TILE_K_HALF>{}),
                make_stride(Int<A_TILE_K_HALF>{}, Int<1>{}))
  ));

  // --- A GMEM -> SMEM TiledCopy (Direction 1) ---
  // Copy atom: cp.async 16 bytes (= 8 half_t) per thread
  // Thread layout: (A_M_THREADS, A_TILE_K_INT4), K-major
  // Value layout: (1, 8) — 8 consecutive half_t along K per access
  using GmemCopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, cute::half_t>;
  using GmemTiledCopyA = decltype(make_tiled_copy(
    GmemCopyAtomA{},
    Layout<Shape<Int<A_M_THREADS>, Int<A_TILE_K_INT4>>,
           Stride<Int<A_TILE_K_INT4>, _1>>{},
    Layout<Shape<_1, _8>>{}
  ));

  // ========================================================================
  // CuTe TiledCopy for B Matrix (Direction 3)
  // ========================================================================

  // B tile dimensions (one pipeline stage)
  // B layout: (K/16, N*16/32) in int4, no swizzle (offline reordered)
  constexpr int B_TILE_K = thread_k_blocks;
  constexpr int B_TILE_N = b_sh_stride;            // = 8 * thread_n_blocks
  constexpr int B_K_THREADS = threads / B_TILE_N;  // threads along K dim

  // B smem layout: flat, no swizzle (data is offline reordered)
  using SmemLayoutB = decltype(make_layout(
    make_shape(Int<B_TILE_K>{}, Int<B_TILE_N>{}),
    make_stride(Int<B_TILE_N>{}, _1{})
  ));

  // B GMEM -> SMEM TiledCopy (cp.async.cg for L2-only caching)
  using GmemCopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, int4>;
  using GmemTiledCopyB = decltype(make_tiled_copy(
    GmemCopyAtomB{},
    Layout<Shape<Int<B_K_THREADS>, Int<B_TILE_N>>,
           Stride<Int<B_TILE_N>, _1>>{},
    Layout<Shape<_1, _1>>{}
  ));

  // --- A SMEM -> Reg via TiledMMA partition (Direction 2 / Step 6.2) ---
  // Use TiledMMA to partition smem A and create register fragments.
  // copy(tCsA, tCrA) produces identical register layout to ldmatrix.
  // Warp layout: (1 on M, N_WARPS_N on N, N_WARPS_K on K)
  constexpr int N_WARPS_N = thread_n_blocks / 4;
  constexpr int N_WARPS_K = (threads / 32) / N_WARPS_N;
  using TiledMma = TiledMMA<
    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
    Layout<Shape<_1, Int<N_WARPS_N>, Int<N_WARPS_K>>>
  >;

  // ========================================================================
  // Per-Thread Index Computation
  // ========================================================================

  // A matrix: K-column base offset for pipeline (replaces a_gl_rd)
  int a_k_col = A_TILE_K_HALF * slice_row;  // in half_t units

  // B matrix: tracked via 2D offsets (replaces B_ptr[] and b_gl_rd)
  int b_k_row = thread_k_blocks * slice_row;  // current K-row offset in B's global layout
  int b_n_col = b_sh_stride * slice_col;       // current N-col offset
  int b_sh_rd = threadIdx.x;                   // smem read index (flat, identity mapping)

  // Scale indices
  int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + s_sh_stride * slice_col + threadIdx.x;
  int s_sh_wr = threadIdx.x;
  int s_sh_rd;
  if (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) % 4;

  // ========================================================================
  // Predication & CuTe Partition Setup for A
  // ========================================================================

  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

  // --- CuTe: instantiate TiledCopy and TiledMMA, get thread slices ---
  GmemTiledCopyA gmem_tiled_copy_a;
  auto gmem_thr_copy_a = gmem_tiled_copy_a.get_thread_slice(threadIdx.x);

  GmemTiledCopyB gmem_tiled_copy_b;
  auto gmem_thr_copy_b = gmem_tiled_copy_b.get_thread_slice(threadIdx.x);

  // TiledMMA for A smem->reg partitioning
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

  // Precompute A gmem->smem predicate
  // Thread tid maps to row = tid / A_TILE_K_INT4 for each M-iteration
  int a_tid_m = threadIdx.x / A_TILE_K_INT4;

  // ========================================================================
  // Shared Memory Allocation
  // ========================================================================

  extern __shared__ int4 sh[];
  int4* sh_a = sh;
  int4* sh_b = sh_a + (stages * a_sh_stage);
  int4* sh_s = sh_b + (stages * b_sh_stage);

  // ========================================================================
  // Register Fragments
  // ========================================================================

  FragA frag_a[2][thread_m_blocks];           // A fragments (double-buffered over k)
  I4    frag_b_quant[2];                      // packed INT4 B fragments (double-buffered)
  FragC frag_c[thread_m_blocks][4][2];        // accumulators [m_block][n_subtile][b_half]
  FragS frag_s[2][4];                         // scale fragments (double-buffered)

  // ========================================================================
  // Lambda: Zero accumulators
  // ========================================================================

  auto zero_accums = [&] () {
    #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  // ========================================================================
  // Lambda: Fetch tile from GMEM to SMEM (cp.async pipeline)
  // ========================================================================

  auto fetch_to_shared = [&] (int pipe, int a_off, bool pred = true) {
    if (pred) {
      // --- A matrix: GMEM -> SMEM via CuTe TiledCopy ---
      // Create gmem tensor for this K-tile (half_t elements)
      auto gA_tile = make_tensor(
        make_gmem_ptr(reinterpret_cast<const cute::half_t*>(cur_A()) + a_k_col + A_TILE_K_HALF * a_off),
        make_shape(Int<A_TILE_M>{}, Int<A_TILE_K_HALF>{}),
        make_stride(prob_k, Int<1>{})
      );
      // Create smem tensor for this stage (half_t with swizzle)
      auto sA_stage = make_tensor(
        make_smem_ptr(reinterpret_cast<cute::half_t*>(sh_a + a_sh_stage * pipe)),
        SmemLayoutA{}
      );
      // Partition source and destination for this thread
      auto tAgA = gmem_thr_copy_a.partition_S(gA_tile);  // (CPY, CPY_M, CPY_K)
      auto tAsA = gmem_thr_copy_a.partition_D(sA_stage); // (CPY, CPY_M, CPY_K)
      // Copy with predication (iterate over M-partitions)
      #pragma unroll
      for (int i = 0; i < size<1>(tAsA); i++) {
        bool m_pred = (a_tid_m + i * A_M_THREADS) < prob_m;
        cp_async4_pred_cute(&tAsA(_0{}, i, _0{}), &tAgA(_0{}, i, _0{}), m_pred);
      }

      // --- B matrix: GMEM -> SMEM via CuTe TiledCopy ---
      {
        auto gB_tile = make_tensor(
          make_gmem_ptr(B + b_gl_stride * b_k_row + b_n_col),
          make_shape(Int<B_TILE_K>{}, Int<B_TILE_N>{}),
          make_stride(b_gl_stride, Int<1>{})
        );
        auto sB_stage = make_tensor(
          make_smem_ptr(sh_b + b_sh_stage * pipe),
          SmemLayoutB{}
        );
        auto tBgB = gmem_thr_copy_b.partition_S(gB_tile);
        auto tBsB = gmem_thr_copy_b.partition_D(sB_stage);
        // No predication needed for B (tile always fully valid)
        #pragma unroll
        for (int i = 0; i < size<1>(tBsB); i++) {
          cp_async4_stream_cute(&tBsB(_0{}, i, _0{}), &tBgB(_0{}, i, _0{}));
        }
        b_k_row += thread_k_blocks;  // advance to next K-tile
      }

      // --- Scales: fetch once per quantization group ---
      if constexpr (group_blocks != -1) {
        if (pipe % (group_blocks / thread_k_blocks) == 0) {
          int4* sh_s_stage = sh_s + s_sh_stage * pipe;
          if (s_sh_wr_pred)
            cp_async4_stream_cute(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);
          s_gl_rd += s_gl_rd_delta;
        }
      }
    }
    cp_async_fence_cute();
  };

  // ========================================================================
  // Lambda: Wait for SMEM stage
  // ========================================================================

  auto wait_for_stage = [&] () {
    cp_async_wait_cute<stages - 2>();
    __syncthreads();
  };

  // ========================================================================
  // Lambda: Fetch from SMEM to registers
  //   A: uses ldmatrix (ldsm4)
  //   B: direct int4 load (then dequant in matmul)
  // ========================================================================

  auto fetch_to_registers = [&] (int k, int pipe) {
    // Load scales if grouped quantization
    if constexpr (group_blocks != -1) {
      int4* sh_s_stage = sh_s + s_sh_stage * ((group_blocks / thread_k_blocks) * (pipe / (group_blocks / thread_k_blocks)));
      reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
    }

    // A: smem -> reg via CuTe TiledMMA partition
    // partition_A gives MMA-compatible smem view, partition_fragment_A gives
    // matching register tensor. copy() auto-vectorizes the transfer.
    {
      auto sA_stage = make_tensor(
        make_smem_ptr(reinterpret_cast<cute::half_t*>(sh_a + a_sh_stage * pipe)),
        SmemLayoutA{}
      );
      auto tCsA = thr_mma.partition_A(sA_stage);           // (MMA=8, MMA_M, MMA_K)

      // Create register fragment backed by frag_a storage
      // partition_fragment_A shape: (MMA=8, MMA_M=m_blocks, MMA_K=b_sh_wr_iters)
      // Each (MMA=8) slice = 8 half_t = 4 half2 = 1 FragA
      int k_subtile = k % b_sh_wr_iters;
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        // Create a register tensor view over frag_a[k%2][i] (8 half_t = 1 FragA)
        auto tCrA_i = make_tensor(
          make_rmem_ptr(reinterpret_cast<cute::half_t*>(&frag_a[k % 2][i])),
          shape(tCsA(_, _0{}, _0{}))  // same shape as MMA mode: (MMA=8)
        );
        copy(tCsA(_, i, k_subtile), tCrA_i);
      }
    }

    // B: direct load of packed INT4 from shared memory
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    frag_b_quant[k % 2] = *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
  };

  // ========================================================================
  // Lambda: MMA computation with dequant-MMA interleaving
  //   Uses m16n8k16 MMA atom. Dequant is interleaved with MMA to hide latency.
  // ========================================================================

  auto matmul = [&] (int k) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      // Dequantize INT4 -> FP16
      int b_quant = frag_b_quant[k % 2][j];
      int b_quant_shift = b_quant >> 8;
      FragB frag_b0 = dequant_cute(b_quant);
      if (group_blocks != -1)
        scale_cute(frag_b0, frag_s[k % 2][j], 0);
      FragB frag_b1 = dequant_cute(b_quant_shift);
      if (group_blocks != -1)
        scale_cute(frag_b1, frag_s[k % 2][j], 1);

      // MMA: iterate over M blocks
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma_cute(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
        mma_cute(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
      }
    }
  };

  // ========================================================================
  // Lambda: Thread-block reduction (warp-level tree reduce in SMEM)
  // ========================================================================

  auto thread_block_reduce = [&] () {
    constexpr int red_off = threads / b_sh_stride / 2;
    if (red_off >= 1) {
      int red_idx = threadIdx.x / b_sh_stride;
      constexpr int red_sh_stride = b_sh_stride * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);

      #pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
        #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
            #pragma unroll
            for (int j = 0; j < 4 * 2; j++) {
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);
                #pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];
              }
              sh[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
          #pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);
            #pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  // ========================================================================
  // Lambda: Global reduction across CTAs (serial in L2)
  // ========================================================================

  auto global_reduce = [&] (bool first = false, bool last = false) {
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 8 * c_gl_stride;
      int c_gl_wr_delta_i = 4 * (active_threads / 32);
      int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) + 4 * (threadIdx.x / 32) + threadIdx.x % 4;
      c_gl_wr += (2 * thread_n_blocks) * slice_col;
      constexpr int c_sh_wr_delta = active_threads;
      int c_sh_wr = threadIdx.x;

      int row = (threadIdx.x % 32) / 4;
      int4* C_cur = cur_C();

      if (!first) {
        #pragma unroll
        for (int i = 0; i < thread_m_blocks * 4; i++) {
          cp_async4_pred_cute(
            &sh[c_sh_wr + c_sh_wr_delta * i],
            &C_cur[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)],
            i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m
          );
        }
        cp_async_fence_cute();
        cp_async_wait_cute<0>();
      }

      #pragma unroll
      for (int i = 0; i < thread_m_blocks * 4; i++) {
        if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {
          if (!first) {
            int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];
            #pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] += __half2float(
                reinterpret_cast<__half*>(&c_red)[j]
              );
            }
          }
          if (!last) {
            int4 c;
            #pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<__half*>(&c)[j] = __float2half(
                reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)]
              );
            }
            C_cur[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] = c;
          }
        }
      }
    }
  };

  // ========================================================================
  // Lambda: Write final result to global C
  // ========================================================================

  auto write_result = [&] () {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;

    // Pack FP32 accumulator -> FP16 half2, optionally apply per-column scale
    auto write = [&] (int idx, float c0, float c1, FragS& s_frag) {
      half2 res = __halves2half2(__float2half(c0), __float2half(c1));
      if (group_blocks == -1)
        res = __hmul2(res, s_frag[0]);
      ((half2*) sh)[idx] = res;
    };

    // Stage 1: Pack frag_c -> shared memory (only active warps)
    if (threadIdx.x / 32 < thread_n_blocks / 4) {
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 8 * j;
          write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1], frag_s[j / 2][2 * (j % 2) + 1]);
          write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3], frag_s[j / 2][2 * (j % 2) + 1]);
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

    // Stage 2: Stream shared -> global C
    int4* C_cur = cur_C();
    #pragma unroll
    for (int i = 0; i < ceildiv_cute(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C_cur[c_gl_wr] = sh[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  };

  // ========================================================================
  // Pipeline Prologue
  // ========================================================================

  auto start_pipes = [&] () {
    zero_accums();
    #pragma unroll
    for (int i = 0; i < stages - 1; i++)
      fetch_to_shared(i, i, i < slice_iters);
    wait_for_stage();
    fetch_to_registers(0, 0);
    a_k_col += A_TILE_K_HALF * (stages - 1);
  };
  start_pipes();

  // ========================================================================
  // Main Loop
  // ========================================================================

  while (slice_iters) {
    #pragma unroll
    for (int pipe = 0; pipe < stages;) {
      #pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0)
        break;
    }
    a_k_col += A_TILE_K_HALF * stages;

    // Post-processing: reduce and possibly move to next column slice
    if (slice_iters == 0) {
      cp_async_wait_cute<0>();
      bool last = slice_idx == slice_count - 1;

      // Per-column scales: fetch in the final step before write-out
      if (group_blocks == -1 && last) {
        if (s_sh_wr_pred)
          cp_async4_stream_cute(&sh_s[s_sh_wr], &s[s_gl_rd]);
        cp_async_fence_cute();
      }

      thread_block_reduce();

      if (group_blocks == -1 && last) {
        cp_async_wait_cute<0>();
        __syncthreads();
        if (threadIdx.x / 32 < thread_n_blocks / 4) {
          reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
          reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
        }
      }

      if (slice_count > 1) {
        barrier_acquire_cute(&cur_locks()[slice_col], slice_idx);
        global_reduce(slice_idx == 0, last);
        barrier_release_cute(&cur_locks()[slice_col], last);
      }
      if (last)
        write_result();

      slice_row = 0;
      slice_col_par++;
      slice_col++;
      init_slice();
      if (slice_iters) {
        a_k_col = 0;      // slice_row is reset to 0 for new slice
        b_k_row = 0;      // K rewinds to 0
        b_n_col = b_sh_stride * slice_col;  // N advances to current slice
        s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
        start_pipes();
      }
    }
  }
}


// ============================================================================
// Host Launch Function
// ============================================================================

const int THREADS_CUTE = 256;
const int STAGES_CUTE = 4;
const int SHARED_MEM_CUTE = 96 * 1024;

#define CALL_IF_CUTE(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS) \
  else if ( \
    thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS && thread_k_blocks == THREAD_K_BLOCKS && \
    group_blocks == GROUP_BLOCKS \
  ) { \
    cudaFuncSetAttribute( \
      MarlinCute<THREADS_CUTE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES_CUTE, GROUP_BLOCKS>, \
      cudaFuncAttributeMaxDynamicSharedMemorySize, \
      SHARED_MEM_CUTE \
    ); \
    MarlinCute< \
      THREADS_CUTE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES_CUTE, GROUP_BLOCKS \
    ><<<blocks, THREADS_CUTE, SHARED_MEM_CUTE, stream>>>( \
      A_ptr, B_ptr, C_ptr, s_ptr, \
      prob_m, prob_n, prob_k, \
      locks \
    ); \
  }


int marlin_cuda_cute(
  const void* A,
  const void* B,
        void* C,
        void* s,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
) {
  int tot_m = prob_m;
  int tot_m_blocks = ceildiv_cute(tot_m, 16);
  int pad = 16 * tot_m_blocks - tot_m;

  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (thread_k == -1 || thread_n == -1) {
    if (prob_m <= 16) {
      thread_k = 128;
      thread_n = 128;
    } else {
      thread_k = 64;
      thread_n = 256;
    }
  }

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;
  int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;
  int blocks = sms;

  if (prob_n % thread_n != 0 || prob_k % thread_k != 0 || (group_blocks != -1 && prob_k % group_blocks != 0))
    return ERR_PROB_SHAPE;
  if (prob_m == 0 || prob_n == 0 || prob_k == 0)
    return 0;

  const int4* A_ptr = (const int4*) A;
  const int4* B_ptr = (const int4*) B;
  int4* C_ptr = (int4*) C;
  const int4* s_ptr = (const int4*) s;

  int* locks = (int*) workspace;

  int ret = 0;
  for (int i = 0; i < tot_m_blocks; i += 4) {
    int thread_m_blocks = tot_m_blocks - i;
    prob_m = tot_m - 16 * i;
    int par = 1;
    if (thread_m_blocks > 4) {
      par = (16 * thread_m_blocks - pad) / 64;
      if (par > max_par)
        par = max_par;
      prob_m = 64 * par;
      i += 4 * (par - 1);
      thread_m_blocks = 4;
    }

    if (false) {}
    CALL_IF_CUTE(1,  8,  8, -1)
    CALL_IF_CUTE(1,  8,  8,  8)
    CALL_IF_CUTE(1, 16,  4, -1)
    CALL_IF_CUTE(1, 16,  4,  8)
    CALL_IF_CUTE(2, 16,  4, -1)
    CALL_IF_CUTE(2, 16,  4,  8)
    CALL_IF_CUTE(3, 16,  4, -1)
    CALL_IF_CUTE(3, 16,  4,  8)
    CALL_IF_CUTE(4, 16,  4, -1)
    CALL_IF_CUTE(4, 16,  4,  8)
    else
      ret = ERR_KERN_SHAPE;

    A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
  }

  return ret;
}
