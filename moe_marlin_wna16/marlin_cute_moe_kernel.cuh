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
// Type Aliases
// ============================================================================

template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;
using FragA = Vec<half2, 4>;   // m16n8k16 A fragment
using FragB = Vec<half2, 2>;   // m16n8k16 B fragment
using FragC = Vec<float, 4>;   // m16n8k16 C fragment
using FragS = Vec<half2, 1>;   // quantization scale fragment

#define ERR_PROB_SHAPE 1
#define ERR_KERN_SHAPE 2

constexpr int ceildiv_cute(int a, int b) {
  return (a + b - 1) / b;
}

// ============================================================================
// PTX Intrinsics — manual PTX for operations without CuTe high-level API
// Placed in namespace to avoid conflicts with original Marlin kernel.
// ============================================================================

namespace marlin_cute {

// Predicated cp.async 16B (for MOE scattered A fetch)
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
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
// MOE CuTe-based Marlin Kernel
// Same GEMM core as W4A16 CuTe, with MOE dispatch wrapper:
//   - Token routing via sorted_token_ids (A row remapping)
//   - Per-expert B/scales offset
//   - topk_weights multiplication in epilog
// ============================================================================

template <
  const int threads,           // threads per CTA (256)
  const int thread_m_blocks,   // M blocks per CTA (1-4, each 16 rows)
  const int thread_n_blocks,   // N blocks per CTA (8 or 16)
  const int thread_k_blocks,   // K blocks per CTA (4 or 8)
  const int stages,            // pipeline stages (4)
  const int group_blocks = -1  // quantization group size in blocks (-1 = per-column)
>
__global__ void MarlinCuteMoe(
  const int4* __restrict__ A,   // FP16 input: (M, K), row-major
  const int4* __restrict__ B,   // INT4 quantized weights: (num_experts, K/16, N*16/32)
        int4* __restrict__ C,   // FP16 output: (M*top_k, N), row-major
  const int4* __restrict__ s,   // FP16 scales: (num_experts, K/groupsize, N/8)
  // --- MOE parameters ---
  const int32_t* __restrict__ sorted_token_ids,     // MOE routed token indices
  const int32_t* __restrict__ expert_ids,            // expert ID per MOE block
  const int32_t* __restrict__ num_tokens_post_padded,// total padded tokens
  const float* __restrict__ topk_weights,            // top-k routing weights
  int top_k,              // number of experts per token
  bool mul_topk_weights,  // whether to multiply by topk weights
  int prob_m,             // original batch size (before top-k expansion)
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

  // MOE: parallel = total MOE blocks (including invalid expert_id=-1 blocks)
  constexpr int moe_block_size = 16 * thread_m_blocks;
  int prob_m_top_k = prob_m * top_k;  // BEFORE overwriting prob_m
  int parallel = num_tokens_post_padded[0] / moe_block_size;
  if (parallel == 0) return;
  prob_m = moe_block_size;

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int total_iters = k_tiles * n_tiles * parallel;
  int iters_per_block = ceildiv_cute(total_iters, gridDim.x);
  // Re-enable split-K (invalid experts handled via no-op barrier participation)
  if (group_blocks != -1)
    iters_per_block = (group_blocks / thread_k_blocks) * ceildiv_cute(iters_per_block, (group_blocks / thread_k_blocks));

  int block_iter_begin = blockIdx.x * iters_per_block;
  int block_iter_end = min(block_iter_begin + iters_per_block, total_iters);
  int remaining_iters = block_iter_end - block_iter_begin;
  if (remaining_iters <= 0) return;  // CTA has no work

  int tile_idx = block_iter_begin / k_tiles;
  TileWorkDesc tile_work;
  tile_work.init(k_tiles, n_tiles, iters_per_block, tile_idx, block_iter_begin, block_iter_end);

  // MOE: per-expert B/scales offset, C offset is per MOE block
  int B_expert_stride = prob_n * prob_k / (8 * 4);  // B size per expert in int4 (INT4 packed)
  int s_expert_stride = prob_n * (group_blocks == -1 ? 1 : prob_k / (16 * group_blocks)) / 8;

  // cur_B/cur_s: offset by expert_id for current MOE block
  auto cur_expert_id = [&]() -> int { return expert_ids[tile_work.m_idx]; };
  auto cur_B = [&]() -> const int4* {
    int eid = cur_expert_id();
    return B + (eid >= 0 ? eid : 0) * B_expert_stride;
  };
  auto cur_s = [&]() -> const int4* {
    int eid = cur_expert_id();
    return s + (eid >= 0 ? eid : 0) * s_expert_stride;
  };
  // A: no per-expert offset (shared input), rows remapped via sorted_token_ids
  // C: output goes to (m_idx * moe_block_size * top_k + row) positions
  auto cur_C = [&]() -> int4* { return C; };  // C is flat (m*top_k, n)
  auto cur_locks = [&]() -> int* { return locks + tile_work.m_idx * n_tiles; };

  // Sorted token IDs for current MOE block (loaded to smem)
  // sh_sorted_ids[row] gives the original token position for this row
  auto cur_sorted_ids = [&]() -> const int32_t* {
    return sorted_token_ids + tile_work.m_idx * moe_block_size;
  };

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

  // --- TiledMMA with MmaPermutations ---
  // MmaPermutations encode Marlin's N iteration order:
  //   _2 (b0,b1 halves) × _4 (j subtiles) × N_WARPS_N × _8 (atom N)
  // This enables make_tiled_copy_C + retile_S for the R2S epilog.
  constexpr int N_WARPS_N = thread_n_blocks / 4;
  constexpr int N_WARPS_K = (threads / 32) / N_WARPS_N;
  using MmaPermuteNLayout = Layout<
    Shape<_2, _4, Int<N_WARPS_N>, _8>,
    Stride<_1, _2, _64, _8>>;
  using MmaPermutations = decltype(make_tile(
    Int<16>{},                   // M = atom_M
    MmaPermuteNLayout{},         // N = full CTA N with dequant interleave
    Int<N_WARPS_K * 16>{}        // K = K per pipeline step
  ));
  using TiledMma = TiledMMA<
    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
    Layout<Shape<_1, Int<N_WARPS_N>, Int<N_WARPS_K>>>,
    MmaPermutations
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
  // C accumulators via partition_fragment_C: shape ((_2,_2), m_blocks, 8)
  // mode 0 = 4 float per MMA, mode 1 = M tiles, mode 2 = N tiles (8 = 4 subtiles × 2 halves)
  auto gC_dummy = make_tensor(make_gmem_ptr((cute::half_t*)nullptr),
    make_shape(Int<16 * thread_m_blocks>{}, Int<16 * thread_n_blocks>{}), LayoutRight{});
  auto tCrC = thr_mma.partition_fragment_C(gC_dummy);
  FragS frag_s[2][4];                         // scale fragments (double-buffered)

  // R2S epilog: make_tiled_copy_C for register→smem shuffle
  using R2SCCopyAtom = Copy_Atom<UniversalCopy<int>, cute::half_t>;
  auto r2s_copy = make_tiled_copy_C(R2SCCopyAtom{}, tiled_mma);
  auto thr_r2s = r2s_copy.get_thread_slice(threadIdx.x);

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
      // --- A matrix: GMEM -> SMEM with MOE token routing ---
      // Rows are scattered via sorted_token_ids — use manual cp.async with CuTe smem layout
      auto sA_stage = make_tensor(
        make_smem_ptr(reinterpret_cast<cute::half_t*>(sh_a + a_sh_stage * pipe)),
        SmemLayoutA{}
      );
      constexpr int A_COPY_THREADS = A_TILE_K_INT4;  // threads per row
      int a_tid_row = threadIdx.x / A_COPY_THREADS;  // which row this thread loads
      int a_tid_col = (threadIdx.x % A_COPY_THREADS) * 8;  // col in half_t (8 per int4)
      constexpr int A_ROWS_PER_ITER = threads / A_COPY_THREADS;
      constexpr int A_ITERS = ceildiv_cute(A_TILE_M, A_ROWS_PER_ITER);
      const int32_t* block_sorted_ids = cur_sorted_ids();
      const cute::half_t* A_half = reinterpret_cast<const cute::half_t*>(A);
      int k_col_offset = a_k_col + A_TILE_K_HALF * a_off;
      #pragma unroll
      for (int i = 0; i < A_ITERS; i++) {
        int row = a_tid_row + i * A_ROWS_PER_ITER;
        if (row < moe_block_size) {
          int32_t sorted_id = block_sorted_ids[row];
          bool valid = sorted_id < prob_m_top_k && (k_col_offset + a_tid_col) < prob_k;
          int32_t actual_row = valid ? sorted_id / top_k : 0;
          // Write to swizzled smem via CuTe layout, read from scattered gmem
          marlin_cute::cp_async4_pred(
            &sA_stage(row, a_tid_col),
            &A_half[actual_row * prob_k + (valid ? k_col_offset + a_tid_col : 0)],
            valid
          );
        }
      }

      // --- B matrix: GMEM -> SMEM via CuTe TiledCopy ---
      {
        auto gB_tile = make_tensor(
          make_gmem_ptr(cur_B() + b_gl_stride * b_k_row + b_n_col),
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
            auto src = make_tensor(make_gmem_ptr(&cur_s()[s_gl_rd]), Int<1>{});
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

  // Epilog smem layout for C: (M, N) in half_t with +8 padding to avoid bank conflicts
  constexpr int CTA_N = 16 * thread_n_blocks;
  using SmemLayoutC = decltype(make_layout(
    make_shape(Int<16 * thread_m_blocks>{}, Int<CTA_N>{}),
    make_stride(Int<CTA_N + 8>{}, _1{})
  ));

  // S2G TiledCopy for C: 256 threads, each copies 8 half_t (uint128_t)
  constexpr int C_S2G_VEC = 8;  // 8 half_t = 16 bytes
  constexpr int C_S2G_N_THREADS = CTA_N / C_S2G_VEC;
  constexpr int C_S2G_M_THREADS = threads / C_S2G_N_THREADS;
  using S2GCCopyAtom = Copy_Atom<UniversalCopy<cute::uint128_t>, cute::half_t>;
  using S2GCCopy = decltype(make_tiled_copy(
    S2GCCopyAtom{},
    make_layout(make_shape(Int<C_S2G_M_THREADS>{}, Int<C_S2G_N_THREADS>{}),
                make_stride(Int<C_S2G_N_THREADS>{}, _1{})),
    Layout<Shape<_1, Int<C_S2G_VEC>>>{}
  ));

  auto write_result = [&] () {
    // Stage 1: FP32 -> FP16 + scale, then R2S via make_tiled_copy_C + retile_S
    // Convert tCrC (FP32) to FP16, applying per-column scale if needed
    // FP32 → FP16 + per-column scale (topk_weights applied in S2G stage)
    auto tCrC_fp16 = make_tensor_like<cute::half_t>(tCrC);
    if (threadIdx.x / 32 < thread_n_blocks / 4) {
      #pragma unroll
      for (int m = 0; m < size<1>(tCrC); m++) {
        #pragma unroll
        for (int n = 0; n < size<2>(tCrC); n++) {
          #pragma unroll
          for (int v = 0; v < size<0>(tCrC); v += 2) {
            half2 res = __halves2half2(__float2half(tCrC(v, m, n)),
                                       __float2half(tCrC(v + 1, m, n)));
            if (group_blocks == -1) {
              int j = n / 2, b = n % 2;
              res = __hmul2(res, frag_s[j / 2][2 * (j % 2) + b][0]);
            }
            reinterpret_cast<half2*>(&tCrC_fp16(v, m, n))[0] = res;
          }
        }
      }
    }

    // R2S: use make_tiled_copy_C + retile_S to handle MMA register shuffle
    // Only N-warps (warp_id < N_WARPS_N) have valid C data after thread_block_reduce
    auto sC = make_tensor(make_smem_ptr(reinterpret_cast<cute::half_t*>(sh)), SmemLayoutC{});
    auto r2s_tCrC = thr_r2s.retile_S(tCrC_fp16);
    auto r2s_tCsC = thr_r2s.partition_D(sC);
    if (threadIdx.x / 32 < thread_n_blocks / 4) {
      copy(r2s_copy, r2s_tCrC, r2s_tCsC);
    }
    __syncthreads();

    // Stage 2: S2G — MOE scattered write to C at sorted_token_ids positions
    // Can't use TiledCopy because rows are non-contiguous in gmem
    // Use simple strided copy: each thread handles some columns
    const int32_t* block_sorted_ids = cur_sorted_ids();
    int4* C_int4 = cur_C();
    int c_gl_stride = prob_n / 8;   // C row stride in int4
    constexpr int C_N_INT4 = CTA_N / 8;  // N-tile width in int4
    // Each thread writes C_N_INT4 / threads * moe_block_size int4 elements
    // Simple scheme: threads tile (M, N_int4) as (threads / C_N_INT4, C_N_INT4)
    constexpr int C_WR_M_THREADS = threads / C_N_INT4;
    int c_tid_n = threadIdx.x % C_N_INT4;
    int c_tid_m = threadIdx.x / C_N_INT4;
    // Read from padded smem, write to scattered gmem
    #pragma unroll
    for (int iter = 0; iter < ceildiv_cute(moe_block_size, C_WR_M_THREADS); iter++) {
      int row = c_tid_m + iter * C_WR_M_THREADS;
      if (row < moe_block_size) {
        int32_t sorted_id = block_sorted_ids[row];
        if (sorted_id < prob_m_top_k) {
          // Read from smem
          int sh_idx = row * (CTA_N + 8) / 8 + c_tid_n;
          int4 val = sh[sh_idx];
          // MOE: multiply by topk_weight if enabled
          if (mul_topk_weights) {
            half2 tw = __float2half2_rn(topk_weights[sorted_id]);
            half2* val_h2 = reinterpret_cast<half2*>(&val);
            #pragma unroll
            for (int i = 0; i < 4; i++)
              val_h2[i] = __hmul2(val_h2[i], tw);
          }
          // Write to gmem at sorted position
          // Use atomicAdd for split-K accumulation (multiple CTAs may write same row)
          int gl_idx = sorted_id * c_gl_stride + C_N_INT4 * tile_work.n_idx + c_tid_n;
          if (tile_work.is_splited()) {
            half2* dst = reinterpret_cast<half2*>(&C_int4[gl_idx]);
            half2* src = reinterpret_cast<half2*>(&val);
            #pragma unroll
            for (int i = 0; i < 4; i++)
              atomicAdd(dst + i, src[i]);
          } else {
            C_int4[gl_idx] = val;
          }
        }
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
  // Always start pipes (zero accums + prefetch). Invalid experts will have
  // zero accums, and A/B reads may go OOB but are predicated/harmless.
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

    // Per-column scales: ALL CTAs need scales for atomicAdd write path
    if (group_blocks == -1) {
      if (s_sh_wr_pred) {
        auto src = make_tensor(make_gmem_ptr(&cur_s()[s_gl_rd]), Int<1>{});
        auto dst = make_tensor(make_smem_ptr(&sh_s[s_sh_wr]), Int<1>{});
        copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, int4>{}, src, dst);
      }
      cute::cp_async_fence();
    }

    thread_block_reduce();

    if (group_blocks == -1) {
      cute::cp_async_wait<0>();
      __syncthreads();
      if (threadIdx.x / 32 < thread_n_blocks / 4) {
        reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
        reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
      }
    }

    // MOE: always write results (use atomicAdd for split-K accumulation)
    // No barrier needed — atomic operations handle concurrent writes
    if (cur_expert_id() >= 0)
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
// MOE Host Launch Function
// ============================================================================

const int THREADS_MOE_CUTE = 256;
const int STAGES_MOE_CUTE = 4;
const int SHARED_MEM_MOE_CUTE = 96 * 1024;

#define CALL_IF_MOE_CUTE(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS) \
  else if ( \
    thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS && thread_k_blocks == THREAD_K_BLOCKS && \
    group_blocks == GROUP_BLOCKS \
  ) { \
    cudaFuncSetAttribute( \
      MarlinCuteMoe<THREADS_MOE_CUTE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES_MOE_CUTE, GROUP_BLOCKS>, \
      cudaFuncAttributeMaxDynamicSharedMemorySize, \
      SHARED_MEM_MOE_CUTE \
    ); \
    MarlinCuteMoe< \
      THREADS_MOE_CUTE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES_MOE_CUTE, GROUP_BLOCKS \
    ><<<blocks, THREADS_MOE_CUTE, SHARED_MEM_MOE_CUTE, stream>>>( \
      A_ptr, B_ptr, C_ptr, s_ptr, \
      sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, \
      top_k, mul_topk_weights, \
      prob_m, prob_n, prob_k, \
      locks \
    ); \
  }


int marlin_cuda_moe_cute(
  const void* A,
  const void* B,
        void* C,
        void* s,
  const int32_t* sorted_token_ids,
  const int32_t* expert_ids,
  const int32_t* num_tokens_post_padded,
  const float* topk_weights,
  int top_k,
  bool mul_topk_weights,
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
  // MOE: moe_block_size = 16 * thread_m_blocks, fixed at 16 for now
  int thread_m_blocks = 1;  // MOE uses single m-block per tile

  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (thread_k == -1 || thread_n == -1) {
    thread_k = 128;
    thread_n = 128;
  }

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;
  int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;
  int blocks = sms;

  if (prob_n % thread_n != 0 || prob_k % thread_k != 0)
    return ERR_PROB_SHAPE;
  if (prob_m == 0 || prob_n == 0 || prob_k == 0)
    return 0;

  const int4* A_ptr = (const int4*) A;
  const int4* B_ptr = (const int4*) B;
  int4* C_ptr = (int4*) C;
  const int4* s_ptr = (const int4*) s;

  int* locks = (int*) workspace;

  int ret = 0;
  if (false) {}
  CALL_IF_MOE_CUTE(1,  8,  8, -1)
  CALL_IF_MOE_CUTE(1,  8,  8,  8)
  CALL_IF_MOE_CUTE(1, 16,  4, -1)
  CALL_IF_MOE_CUTE(1, 16,  4,  8)
  else
    ret = ERR_KERN_SHAPE;

  return ret;
}
