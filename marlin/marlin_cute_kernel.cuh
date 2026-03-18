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
// PTX Intrinsics — manual PTX for operations without CuTe high-level API
// Placed in namespace to avoid conflicts with original Marlin kernel.
// ============================================================================

namespace marlin_cute {

// lop3: 3-input logical operation (used by INT4 dequantization)
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}

// INT4 dequantization: int32 -> CuTe tensor of 2x half2 (= 4x FP16)
__device__ inline auto dequant_4bit(int q) {
  using namespace cute;
  auto frag = make_tensor<half2>(make_shape(_2{}));
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  frag(0) = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag(1) = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );
  return frag;
}

// Scale a dequantized B fragment (CuTe tensor<half2, (_2,)>)
template <class FragTensor>
__device__ inline void scale_frag(FragTensor& frag, FragS& frag_s, int i) {
  half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
  frag(0) = __hmul2(frag(0), s);
  frag(1) = __hmul2(frag(1), s);
}

// Cross-CTA synchronization barriers (Marlin-specific)
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

__device__ inline void barrier_release(int* lock, bool reset = false) {
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

} // namespace marlin_cute

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
  // CTA Dispatcher — TileWorkDesc
  // Clean tile-based work description replacing scattered slice_* variables.
  // Each CTA gets a contiguous range of K-iterations [block_iter_begin, end).
  // A tile = one (m_idx, n_idx) output tile requiring a full K reduction.
  // ========================================================================

  struct TileWorkDesc {
    int tile_idx;       // flat index in (m_parallel * n_tiles) space
    int m_idx, n_idx;   // 2D tile coordinate
    int k_iter_begin;   // K start within tile (0 = tile start)
    int k_iter_end;     // K end within tile
    int k_iters_remaining;
    int slice_count;    // CTAs contributing to this tile
    int slice_idx;      // this CTA's rank (0 = goes first in barrier = highest K)

    __device__ void init(int k_tiles, int n_tiles, int iters_per_block,
                         int tile_idx_, int block_begin, int block_end) {
      tile_idx = tile_idx_;
      m_idx = tile_idx / n_tiles;
      n_idx = tile_idx % n_tiles;
      int tile_k_begin = tile_idx * k_tiles;
      int tile_k_end = tile_k_begin + k_tiles;
      k_iter_begin = max(block_begin, tile_k_begin) - tile_k_begin;
      k_iter_end = min(block_end, tile_k_end) - tile_k_begin;
      k_iters_remaining = max(0, k_iter_end - k_iter_begin);

      // Compute slice_count and slice_idx (matching original Marlin ordering)
      // slice_idx=0: highest K (goes first in barrier), slice_idx=count-1: lowest K (last)
      int first_block = tile_k_begin / iters_per_block;
      int last_block_global = (tile_k_end - 1) / iters_per_block;
      slice_count = last_block_global - first_block + 1;
      // Reverse: highest block (highest K) → slice_idx=0
      slice_idx = last_block_global - (block_begin / iters_per_block);
    }
    __device__ bool is_first()   const { return slice_idx == 0; }
    __device__ bool is_last()    const { return slice_idx == slice_count - 1; }
    __device__ bool is_splited() const { return slice_count > 1; }
  };

  int parallel = 1;
  if (prob_m > 16 * thread_m_blocks) {
    parallel = prob_m / (16 * thread_m_blocks);
    prob_m = 16 * thread_m_blocks;
  }

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int total_iters = k_tiles * n_tiles * parallel;
  int iters_per_block = ceildiv_cute(total_iters, gridDim.x);
  if (group_blocks != -1)
    iters_per_block = (group_blocks / thread_k_blocks) * ceildiv_cute(iters_per_block, (group_blocks / thread_k_blocks));

  int block_iter_begin = blockIdx.x * iters_per_block;
  int block_iter_end = min(block_iter_begin + iters_per_block, total_iters);
  int remaining_iters = block_iter_end - block_iter_begin;
  if (remaining_iters <= 0) return;  // CTA has no work

  int tile_idx = block_iter_begin / k_tiles;
  TileWorkDesc tile_work;
  tile_work.init(k_tiles, n_tiles, iters_per_block, tile_idx, block_iter_begin, block_iter_end);

  // Compute current A/C/locks from tile coordinate
  auto cur_A = [&]() -> const int4* { return A + tile_work.m_idx * 16 * thread_m_blocks * prob_k / 8; };
  auto cur_C = [&]() -> int4* { return C + tile_work.m_idx * 16 * thread_m_blocks * prob_n / 8; };
  auto cur_locks = [&]() -> int* { return locks + tile_work.m_idx * n_tiles; };

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
  using GmemCopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::half_t>;
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

  // --- A SMEM -> Reg via TiledMMA + ldmatrix (Step 6.2) ---
  // Standard CuTe pattern: make_tiled_copy_A + retile_D bridges
  // ldmatrix copy atom with MMA-compatible register layout.
  constexpr int N_WARPS_N = thread_n_blocks / 4;
  constexpr int N_WARPS_K = (threads / 32) / N_WARPS_N;
  using TiledMma = TiledMMA<
    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
    Layout<Shape<_1, Int<N_WARPS_N>, Int<N_WARPS_K>>>
  >;
  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>;

  // ========================================================================
  // Per-Thread Index Computation (derived from tile_work)
  // ========================================================================

  // Data offsets computed from tile_work, updated when tile changes
  int a_k_col = A_TILE_K_HALF * tile_work.k_iter_begin;
  int b_k_row = thread_k_blocks * tile_work.k_iter_begin;
  int b_n_col = b_sh_stride * tile_work.n_idx;
  int b_sh_rd = threadIdx.x;
  int s_gl_rd;
  if constexpr (group_blocks != -1)
    s_gl_rd = s_gl_stride * ((thread_k_blocks * tile_work.k_iter_begin) / group_blocks) + s_sh_stride * tile_work.n_idx + threadIdx.x;
  else
    s_gl_rd = s_sh_stride * tile_work.n_idx + threadIdx.x;
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

  // TiledMMA + ldmatrix-compatible smem->reg copy for A
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  auto smem_tiled_copy_a = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
  auto smem_thr_copy_a = smem_tiled_copy_a.get_thread_slice(threadIdx.x);


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

  // A fragments: CuTe register tensors (double-buffered), allocated once
  // Shape: (MMA=8 half_t, MMA_M=m_blocks, MMA_K=b_sh_wr_iters)
  auto sA_dummy = make_tensor(make_smem_ptr((cute::half_t*)nullptr), SmemLayoutA{});
  auto tCrA_buf0 = thr_mma.partition_fragment_A(sA_dummy);
  auto tCrA_buf1 = thr_mma.partition_fragment_A(sA_dummy);

  I4    frag_b_quant[2];                      // packed INT4 B fragments (double-buffered)
  // C accumulators: CuTe register tensor (4 float per MMA, m_blocks M, 8 N-subtiles)
  // Layout: (v=4, m=m_blocks, n=8) with strides (1, 32, 4) to match original frag_c[m][4][2]
  // Access: tCrC(v, m_block, j*2+b) ↔ old frag_c[m_block][j][b].elems[v]
  auto tCrC = make_tensor<float>(
    make_shape(_4{}, Int<thread_m_blocks>{}, _8{}),
    make_stride(_1{}, Int<32>{}, _4{})
  );
  FragS frag_s[2][4];                         // scale fragments (double-buffered)

  // ========================================================================
  // Lambda: Zero accumulators
  // ========================================================================

  auto zero_accums = [&] () {
    clear(tCrC);
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
      // Partition source, destination, and predicate via identity tensor
      auto tAgA = gmem_thr_copy_a.partition_S(gA_tile);
      auto tAsA = gmem_thr_copy_a.partition_D(sA_stage);
      auto gA_identity = make_identity_tensor(shape(gA_tile));
      auto tAgA_id = gmem_thr_copy_a.partition_S(gA_identity);
      // M-bounds predication: CuTe identity tensor for coordinate lookup
      #pragma unroll
      for (int m = 0; m < size<1>(tAsA); m++) {
        if (get<0>(tAgA_id(_0{}, m, _0{})) < prob_m) {
          #pragma unroll
          for (int k = 0; k < size<2>(tAsA); k++)
            copy(gmem_tiled_copy_a, tAgA(_, m, k), tAsA(_, m, k));
        }
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
        // No predication needed for B — use CuTe copy directly (cp.async.cg)
        copy(gmem_tiled_copy_b, tBgB, tBsB);
        b_k_row += thread_k_blocks;  // advance to next K-tile
      }

      // --- Scales: fetch once per quantization group via CuTe cp.async ---
      if constexpr (group_blocks != -1) {
        if (pipe % (group_blocks / thread_k_blocks) == 0) {
          int4* sh_s_stage = sh_s + s_sh_stage * pipe;
          if (s_sh_wr_pred) {
            auto src = make_tensor(make_gmem_ptr(&s[s_gl_rd]), Int<1>{});
            auto dst = make_tensor(make_smem_ptr(&sh_s_stage[s_sh_wr]), Int<1>{});
            copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, int4>{}, src, dst);
          }
          s_gl_rd += s_gl_rd_delta;
        }
      }
    }
    cute::cp_async_fence();
  };

  // ========================================================================
  // Lambda: Wait for SMEM stage
  // ========================================================================

  auto wait_for_stage = [&] () {
    cute::cp_async_wait<stages - 2>();
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

    // A: smem -> reg via make_tiled_copy_A + retile_D (ldmatrix)
    // Loads directly into persistent tCrA buffers (no intermediate copy)
    {
      auto sA_stage = make_tensor(
        make_smem_ptr(reinterpret_cast<cute::half_t*>(sh_a + a_sh_stage * pipe)),
        SmemLayoutA{}
      );
      auto tCsA = smem_thr_copy_a.partition_S(sA_stage);
      auto& tCrA = (k % 2 == 0) ? tCrA_buf0 : tCrA_buf1;
      auto tCrA_copy = smem_thr_copy_a.retile_D(tCrA);
      int k_subtile = k % b_sh_wr_iters;
      copy(smem_tiled_copy_a, tCsA(_, _, k_subtile), tCrA_copy(_, _, k_subtile));
    }

    // B: direct load of packed INT4 from shared memory
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    frag_b_quant[k % 2] = *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
  };

  // ========================================================================
  // Lambda: MMA computation with dequant-MMA interleaving
  //   Uses m16n8k16 MMA atom. Dequant is interleaved with MMA to hide latency.
  // ========================================================================

  // MMA atom for direct gemm() calls
  using MmaOp = cute::SM80_16x8x16_F32F16F16F32_TN;
  MMA_Atom<MmaOp> mma_atom;

  auto matmul = [&] (int k) {
    auto& tCrA = (k % 2 == 0) ? tCrA_buf0 : tCrA_buf1;
    int k_subtile = k % b_sh_wr_iters;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      // Dequantize INT4 -> CuTe tensor<half2, (_2,)>
      int b_quant = frag_b_quant[k % 2][j];
      auto dq_b0 = marlin_cute::dequant_4bit(b_quant);
      auto dq_b1 = marlin_cute::dequant_4bit(b_quant >> 8);
      if (group_blocks != -1) {
        marlin_cute::scale_frag(dq_b0, frag_s[k % 2][j], 0);
        marlin_cute::scale_frag(dq_b1, frag_s[k % 2][j], 1);
      }
      // recast half2 -> half_t for MMA B operand (4 half_t = 2 uint32)
      auto b0 = recast<cute::half_t>(dq_b0);
      auto b1 = recast<cute::half_t>(dq_b1);

      // MMA via CuTe gemm: C += A * B
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        gemm(mma_atom, tCrA(_, i, k_subtile), b0, tCrC(_, i, j * 2));
        gemm(mma_atom, tCrA(_, i, k_subtile), b1, tCrC(_, i, j * 2 + 1));
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
                  tCrC(k, m_block, j) += c_rd[k] + c_wr[k];
              }
              sh[red_sh_wr] = *reinterpret_cast<const int4*>(&tCrC(0, m_block, j));
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
              tCrC(j, m_block, i) += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  // ========================================================================
  // Lambda: Global reduction across CTAs (serial in L2)
  // ========================================================================

  // Global reduce smem layout: (thread_m_blocks*4, active_threads) in int4
  // Each thread stores/loads one int4 (8 fp16) per M-iteration
  constexpr int GR_ACTIVE_THREADS = 32 * thread_n_blocks / 4;

  auto global_reduce = [&] (bool first = false, bool last = false) {
    if (threadIdx.x < GR_ACTIVE_THREADS) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 8 * c_gl_stride;
      int c_gl_wr_delta_i = 4 * (GR_ACTIVE_THREADS / 32);
      int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) + 4 * (threadIdx.x / 32) + threadIdx.x % 4;
      c_gl_wr += (2 * thread_n_blocks) * tile_work.n_idx;
      int row = (threadIdx.x % 32) / 4;
      int4* C_cur = cur_C();

      // Smem for partial sums: flat, each thread has stride=active_threads
      auto sC_reduce = make_tensor(
        make_smem_ptr(sh + threadIdx.x),
        make_shape(Int<thread_m_blocks * 4>{}),
        make_stride(Int<GR_ACTIVE_THREADS>{})
      );

      if (!first) {
        // Load C partial sums from gmem to smem via cp.async
        #pragma unroll
        for (int i = 0; i < thread_m_blocks * 4; i++) {
          if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {
            auto src = make_tensor(make_gmem_ptr(&C_cur[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)]), Int<1>{});
            auto dst = make_tensor(make_smem_ptr(&sC_reduce(i)), Int<1>{});
            copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, int4>{}, src, dst);
          }
        }
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
      }

      #pragma unroll
      for (int i = 0; i < thread_m_blocks * 4; i++) {
        if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {
          // Accumulate from smem partial sum
          if (!first) {
            int4 c_red = sC_reduce(i);
            #pragma unroll
            for (int j = 0; j < 2 * 4; j++)
              tCrC(i % 4, i / 4, j) += __half2float(reinterpret_cast<__half*>(&c_red)[j]);
          }
          // Write current partial sum to gmem
          if (!last) {
            int4 c;
            #pragma unroll
            for (int j = 0; j < 2 * 4; j++)
              reinterpret_cast<__half*>(&c)[j] = __float2half(tCrC(i % 4, i / 4, j));
            C_cur[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] = c;
          }
        }
      }
    }
  };

  // ========================================================================
  // Lambda: Write final result to global C
  // ========================================================================

  // Epilog smem layout for C: (M, N) in int4 with +1 padding to avoid bank conflicts
  // Same as original c_sh_stride = 2 * thread_n_blocks + 1
  constexpr int C_SH_N = 2 * thread_n_blocks;     // N-tiles in int4
  constexpr int C_SH_STRIDE = C_SH_N + 1;         // padded stride
  using SmemLayoutC = decltype(make_layout(
    make_shape(Int<16 * thread_m_blocks>{}, Int<C_SH_N>{}),
    make_stride(Int<C_SH_STRIDE>{}, _1{})
  ));

  // S2G TiledCopy for C: 256 threads, each copies 1 int4 (8 fp16)
  constexpr int C_S2G_N_THREADS = C_SH_N;
  constexpr int C_S2G_M_THREADS = threads / C_S2G_N_THREADS;
  using S2GCCopyAtom = Copy_Atom<UniversalCopy<cute::uint128_t>, int4>;
  using S2GCCopy = decltype(make_tiled_copy(
    S2GCCopyAtom{},
    make_layout(make_shape(Int<C_S2G_M_THREADS>{}, Int<C_S2G_N_THREADS>{}),
                make_stride(Int<C_S2G_N_THREADS>{}, _1{})),
    Layout<Shape<_1, _1>>{}
  ));

  auto write_result = [&] () {
    // Stage 1: Pack frag_c (FP32) -> shared memory (FP16) — manual MMA-specific shuffle
    // Only active warps (those assigned to N) participate
    constexpr int c_sh_stride = C_SH_STRIDE;
    int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);

    auto write = [&] (int idx, float c0, float c1, FragS& s_frag) {
      half2 res = __halves2half2(__float2half(c0), __float2half(c1));
      if (group_blocks == -1)
        res = __hmul2(res, s_frag[0]);
      ((half2*) sh)[idx] = res;
    };

    if (threadIdx.x / 32 < thread_n_blocks / 4) {
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 8 * j;
          write(wr + (4 * c_sh_stride) * 0 + 0, tCrC(0, i, j*2), tCrC(1, i, j*2), frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 8 + 0, tCrC(2, i, j*2), tCrC(3, i, j*2), frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 0 + 4, tCrC(0, i, j*2+1), tCrC(1, i, j*2+1), frag_s[j / 2][2 * (j % 2) + 1]);
          write(wr + (4 * c_sh_stride) * 8 + 4, tCrC(2, i, j*2+1), tCrC(3, i, j*2+1), frag_s[j / 2][2 * (j % 2) + 1]);
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

    // Stage 2: Stream shared -> global C via CuTe TiledCopy
    auto sC = make_tensor(make_smem_ptr(sh), SmemLayoutC{});
    // Global C tile for this CTA's output
    int4* C_cur = cur_C();
    int c_gl_stride = prob_n / 8;
    auto gC_tile = make_tensor(
      make_gmem_ptr(C_cur + C_SH_N * tile_work.n_idx),
      make_shape(Int<16 * thread_m_blocks>{}, Int<C_SH_N>{}),
      make_stride(c_gl_stride, _1{})
    );

    S2GCCopy s2g_c_copy;
    auto thr_s2g = s2g_c_copy.get_thread_slice(threadIdx.x);
    auto tCsC = thr_s2g.partition_S(sC);
    auto tCgC = thr_s2g.partition_D(gC_tile);
    // Predication via identity tensor for M-bounds
    auto gC_identity = make_identity_tensor(shape(gC_tile));
    auto tCgC_id = thr_s2g.partition_D(gC_identity);
    #pragma unroll
    for (int m = 0; m < size<1>(tCgC); m++) {
      if (get<0>(tCgC_id(_0{}, m, _0{})) < prob_m) {
        #pragma unroll
        for (int n = 0; n < size<2>(tCgC); n++)
          copy(s2g_c_copy, tCsC(_, m, n), tCgC(_, m, n));
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
      fetch_to_shared(i, i, i < tile_work.k_iters_remaining);
    wait_for_stage();
    fetch_to_registers(0, 0);
    a_k_col += A_TILE_K_HALF * (stages - 1);
  };
  start_pipes();

  // ========================================================================
  // Main Loop
  // ========================================================================

  #pragma unroll 1
  while (true) {
    remaining_iters -= tile_work.k_iters_remaining;

    // Main K loop for current tile
    #pragma unroll 1
    while (tile_work.k_iters_remaining) {
      #pragma unroll
      for (int pipe = 0; pipe < stages;) {
        #pragma unroll
        for (int k = 0; k < b_sh_wr_iters; k++) {
          fetch_to_registers(k + 1, pipe % stages);
          if (k == b_sh_wr_iters - 2) {
            fetch_to_shared((pipe + stages - 1) % stages, pipe, tile_work.k_iters_remaining >= stages);
            pipe++;
            wait_for_stage();
          }
          matmul(k);
        }
        tile_work.k_iters_remaining--;
        if (tile_work.k_iters_remaining == 0)
          break;
      }
      a_k_col += A_TILE_K_HALF * stages;
    }

    // ---- Epilog for current tile ----
    cute::cp_async_wait<0>();

    // Per-column scales: fetch in the final step before write-out
    if (group_blocks == -1 && tile_work.is_last()) {
      if (s_sh_wr_pred) {
        auto src = make_tensor(make_gmem_ptr(&s[s_gl_rd]), Int<1>{});
        auto dst = make_tensor(make_smem_ptr(&sh_s[s_sh_wr]), Int<1>{});
        copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, int4>{}, src, dst);
      }
      cute::cp_async_fence();
    }

    thread_block_reduce();

    if (group_blocks == -1 && tile_work.is_last()) {
      cute::cp_async_wait<0>();
      __syncthreads();
      if (threadIdx.x / 32 < thread_n_blocks / 4) {
        reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
        reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
      }
    }

    // Inter-CTA global reduce (only when tile is split across CTAs)
    // slice_idx=0 (highest K) goes first → no wait on initial lock=0
    if (tile_work.is_splited()) {
      marlin_cute::barrier_acquire(&cur_locks()[tile_work.n_idx], tile_work.slice_idx);
      global_reduce(tile_work.is_first(), tile_work.is_last());
      marlin_cute::barrier_release(&cur_locks()[tile_work.n_idx], tile_work.is_last());
    }
    if (tile_work.is_last())
      write_result();

    if (remaining_iters <= 0)
      break;

    // ---- Move to next tile ----
    tile_idx++;
    tile_work.init(k_tiles, n_tiles, iters_per_block, tile_idx, block_iter_begin, block_iter_end);
    a_k_col = A_TILE_K_HALF * tile_work.k_iter_begin;
    b_k_row = thread_k_blocks * tile_work.k_iter_begin;
    b_n_col = b_sh_stride * tile_work.n_idx;
    if constexpr (group_blocks != -1)
      s_gl_rd = s_gl_stride * ((thread_k_blocks * tile_work.k_iter_begin) / group_blocks)
              + s_sh_stride * tile_work.n_idx + threadIdx.x;
    else
      s_gl_rd = s_sh_stride * tile_work.n_idx + threadIdx.x;
    start_pipes();
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
