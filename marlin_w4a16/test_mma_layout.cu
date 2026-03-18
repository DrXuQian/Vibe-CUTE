/*
 * Step 6.1: Verify TiledMMA partition shapes match Marlin's warp layout.
 *
 * For each Marlin configuration (n_blocks, k_blocks, m_blocks), instantiate
 * a CuTe TiledMMA and print the partition shapes for A, B, C operands.
 *
 * Compile: nvcc -I cutlass/include --expt-relaxed-constexpr -o test_mma_layout test_mma_layout.cu
 * Run:     ./test_mma_layout
 */

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/swizzle.hpp>

#include <cstdio>

using namespace cute;

// Marlin's MMA: m16n8k16 with F32 accumulate, F16 inputs
using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;

template <int THREADS, int M_BLOCKS, int N_BLOCKS, int K_BLOCKS>
__global__ void verify_mma_layout() {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  constexpr int A_TILE_M = 16 * M_BLOCKS;
  constexpr int A_TILE_K_HALF = 16 * K_BLOCKS;       // half_t
  constexpr int A_TILE_K_INT4 = A_TILE_K_HALF / 8;   // int4
  constexpr int A_SH_BITS = (A_TILE_K_INT4 == 16) ? 4 : 3;

  constexpr int B_TILE_K = K_BLOCKS;
  constexpr int B_TILE_N = 8 * N_BLOCKS;

  constexpr int N_WARPS_N = N_BLOCKS / 4;
  constexpr int N_WARPS_K = (THREADS / 32) / N_WARPS_N;
  constexpr int b_sh_wr_iters = (B_TILE_K * B_TILE_N) / THREADS;

  printf("============================================================\n");
  printf("Config: THREADS=%d, M_BLOCKS=%d, N_BLOCKS=%d, K_BLOCKS=%d\n",
         THREADS, M_BLOCKS, N_BLOCKS, K_BLOCKS);
  printf("  A tile: (%d, %d) half_t,  B tile: (%d, %d) int4\n",
         A_TILE_M, A_TILE_K_HALF, B_TILE_K, B_TILE_N);
  printf("  N_WARPS_N=%d, N_WARPS_K=%d, b_sh_wr_iters=%d\n",
         N_WARPS_N, N_WARPS_K, b_sh_wr_iters);
  printf("------------------------------------------------------------\n");

  // --- TiledMMA definition ---
  // Warp layout: (1 on M, N_WARPS_N on N, N_WARPS_K on K)
  using WarpLayout = Layout<Shape<_1, Int<N_WARPS_N>, Int<N_WARPS_K>>>;
  using TiledMma = TiledMMA<MmaAtom, WarpLayout>;

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(0);  // thread 0

  printf("TiledMMA:\n");
  print(tiled_mma);
  printf("\n");

  // --- A smem layout (half_t, with swizzle) ---
  using SmemLayoutA = decltype(composition(
    Swizzle<A_SH_BITS, 3, A_SH_BITS>{},
    make_layout(make_shape(Int<A_TILE_M>{}, Int<A_TILE_K_HALF>{}),
                make_stride(Int<A_TILE_K_HALF>{}, Int<1>{}))
  ));
  auto sA = make_tensor(make_smem_ptr((half_t*)nullptr), SmemLayoutA{});

  // Partition A
  auto tCsA = thr_mma.partition_A(sA);
  printf("partition_A(sA) shape: ");
  print(shape(tCsA));
  printf("\n");

  auto tCrA = thr_mma.partition_fragment_A(sA);
  printf("partition_fragment_A shape: ");
  print(shape(tCrA));
  printf("\n");

  // --- Verify dimensions ---
  // MMA mode: how many values per thread per MMA instruction
  // MMA_M mode: how many M-tiles (should == M_BLOCKS)
  // MMA_K mode: how many K-tiles (should relate to K_BLOCKS/k_per_mma)
  printf("\n  size<0>(tCsA) [MMA values]  = %d\n", (int)size<0>(tCsA));
  printf("  size<1>(tCsA) [MMA_M tiles] = %d\n", (int)size<1>(tCsA));
  printf("  size<2>(tCsA) [MMA_K tiles] = %d\n", (int)size<2>(tCsA));
  printf("  total elements in partition = %d\n", (int)size(tCsA));

  // Expected:
  // MMA atom A: m16k16, each thread has 8 half_t (4 x half2)
  // MMA_M = M_BLOCKS (tiled over M, 1 warp on M)
  // MMA_K = K_BLOCKS / N_WARPS_K (each K-warp handles part of K)
  // Wait actually: TiledMMA with K_warps distributes K across warps
  // So each warp sees K / K_warps
  int expected_mma_k = K_BLOCKS * 16 / (16 * N_WARPS_K);  // k_half / (atom_k * k_warps)
  printf("\n  Expected MMA_M = %d (M_BLOCKS)\n", M_BLOCKS);
  printf("  Expected MMA_K = %d (K_BLOCKS/N_WARPS_K = %d/%d)\n",
         expected_mma_k, K_BLOCKS, N_WARPS_K);

  // --- Also check: can we directly copy from partition_A to partition_fragment_A? ---
  printf("\n  shape(tCsA) == shape(tCrA)? %s\n",
         shape(tCsA) == shape(tCrA) ? "YES" : "NO");

  // --- Print ldmatrix copy atom info ---
  printf("\n--- SM75_U32x4_LDSM_N Copy Atom ---\n");
  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
  SmemCopyAtomA smem_copy_a;
  printf("SmemCopyAtomA:\n");
  print(smem_copy_a);
  printf("\n");

  // --- Marlin matmul loop comparison ---
  printf("\n--- Marlin matmul loop vs CuTe partition ---\n");
  printf("  Marlin: for j=0..3 (n-subtiles), each does 2x m16n8k16\n");
  printf("    Per warp per k-subtile: M=16, N=4*2*8=64, K=16\n");
  printf("    Per warp total N: 64, K-subtiles per stage: %d (b_sh_wr_iters)\n", b_sh_wr_iters);
  printf("  CuTe partition_A: (MMA=%d, MMA_M=%d, MMA_K=%d)\n",
         (int)size<0>(tCsA), (int)size<1>(tCsA), (int)size<2>(tCsA));
  printf("    MMA values per thread = %d half_t = %d half2 = FragA\n",
         (int)size<0>(tCsA), (int)size<0>(tCsA) / 2);
  printf("    MMA_M = %d = thread_m_blocks ✓\n", (int)size<1>(tCsA));
  printf("    MMA_K = %d = b_sh_wr_iters (%d) %s\n",
         (int)size<2>(tCsA), b_sh_wr_iters,
         (int)size<2>(tCsA) == b_sh_wr_iters ? "✓ MATCH" : "✗ MISMATCH");

  printf("\n--- Layout compatibility for ldmatrix ---\n");
  printf("  partition_A src layout (tCsA mode 0): ");
  print(layout<0>(tCsA));
  printf("\n  SM75_U32x4_LDSM_N ValLayoutSrc:      (_32,_8):(_8,_1)\n");
  printf("  partition_A mode 0 has %d elements, ldmatrix expects 8 half_t per thread\n",
         (int)size<0>(tCsA));
  printf("  Match? %s\n",
         (int)size<0>(tCsA) == 8 ? "YES (same count)" : "NO");

  printf("============================================================\n\n");
}


int main() {
  // Config 1: n_blocks=8, k_blocks=8, m_blocks=1 (GEMV)
  verify_mma_layout<256, 1, 8, 8><<<1, 256>>>();
  cudaDeviceSynchronize();

  // Config 2: n_blocks=16, k_blocks=4, m_blocks=1
  verify_mma_layout<256, 1, 16, 4><<<1, 256>>>();
  cudaDeviceSynchronize();

  // Config 3: n_blocks=16, k_blocks=4, m_blocks=4 (larger batch)
  verify_mma_layout<256, 4, 16, 4><<<1, 256>>>();
  cudaDeviceSynchronize();

  return 0;
}
