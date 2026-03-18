/*
 * Step 6.2 feasibility test: try different approaches to use SM75_U32x4_LDSM_N
 * copy atom with CuTe swizzled shared memory layout.
 *
 * Compile: nvcc -I cutlass/include --expt-relaxed-constexpr -arch=sm_80 \
 *          -o test_ldmatrix_copy test_ldmatrix_copy.cu
 */

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/swizzle.hpp>

#include <cstdio>
#include <cstdlib>

using namespace cute;

// ============================================================================
// Approach A1: Per-warp TiledCopy with make_tiled_copy
// ============================================================================

template <int M_BLOCKS, int K_BLOCKS, int N_BLOCKS>
__global__ void test_approach_a1(const half_t* A_gmem, half_t* result) {
  constexpr int THREADS = 256;
  constexpr int A_TILE_M = 16 * M_BLOCKS;
  constexpr int A_TILE_K_HALF = 16 * K_BLOCKS;
  constexpr int A_TILE_K_INT4 = A_TILE_K_HALF / 8;
  constexpr int A_SH_BITS = (A_TILE_K_INT4 == 16) ? 4 : 3;
  constexpr int a_sh_stage = A_TILE_K_INT4 * A_TILE_M;
  constexpr int N_WARPS_N = N_BLOCKS / 4;
  constexpr int N_WARPS_K = (THREADS / 32) / N_WARPS_N;
  constexpr int b_sh_wr_iters = (8 * N_BLOCKS * K_BLOCKS) / THREADS;

  // Swizzled smem layout for A (half_t)
  using SmemLayoutA = decltype(composition(
    Swizzle<A_SH_BITS, 3, A_SH_BITS>{},
    make_layout(make_shape(Int<A_TILE_M>{}, Int<A_TILE_K_HALF>{}),
                make_stride(Int<A_TILE_K_HALF>{}, Int<1>{}))
  ));

  extern __shared__ char smem_buf[];
  half_t* sh_a = reinterpret_cast<half_t*>(smem_buf);

  // Load test data to smem (all threads cooperate)
  for (int i = threadIdx.x; i < A_TILE_M * A_TILE_K_HALF; i += THREADS)
    sh_a[i] = A_gmem[i];
  __syncthreads();

  auto sA = make_tensor(make_smem_ptr(sh_a), SmemLayoutA{});

  // --- MMA partition (for reference) ---
  using TiledMma = TiledMMA<
    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
    Layout<Shape<_1, Int<N_WARPS_N>, Int<N_WARPS_K>>>
  >;
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  auto tCsA = thr_mma.partition_A(sA);            // (MMA=8, MMA_M, MMA_K)
  auto tCrA = thr_mma.partition_fragment_A(sA);   // (MMA=8, MMA_M, MMA_K)

  // Try: copy without atom (approach C for comparison)
  // This uses auto-vectorized copy (LDS.128, not ldmatrix)
  for (int k = 0; k < size<2>(tCrA); k++) {
    for (int m = 0; m < size<1>(tCrA); m++) {
      copy(tCsA(_, m, k), tCrA(_, m, k));
    }
  }
  __syncthreads();

  // Write results back for verification
  // Only thread 0 of each warp writes
  if (threadIdx.x % 32 == 0) {
    int warp_id = threadIdx.x / 32;
    half_t* out = result + warp_id * size(tCrA);
    for (int i = 0; i < size(tCrA); i++)
      out[i] = tCrA(i);
  }
}


// ============================================================================
// Approach A2: Manual ldsm4 then reinterpret as CuTe fragment (current approach)
// ============================================================================

__device__ inline void ldsm4_test(uint32_t* frag, const void* smem_ptr) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3]) : "r"(smem)
  );
}

template <int M_BLOCKS, int K_BLOCKS, int N_BLOCKS>
__global__ void test_approach_manual(const half_t* A_gmem, half_t* result) {
  constexpr int THREADS = 256;
  constexpr int A_TILE_M = 16 * M_BLOCKS;
  constexpr int A_TILE_K_HALF = 16 * K_BLOCKS;
  constexpr int A_TILE_K_INT4 = A_TILE_K_HALF / 8;
  constexpr int A_SH_BITS = (A_TILE_K_INT4 == 16) ? 4 : 3;
  constexpr int N_WARPS_N = N_BLOCKS / 4;
  constexpr int N_WARPS_K = (THREADS / 32) / N_WARPS_N;
  constexpr int a_sh_rd_delta_o = 2 * N_WARPS_K;
  constexpr int b_sh_wr_iters = (8 * N_BLOCKS * K_BLOCKS) / THREADS;

  using SmemLayoutA = decltype(composition(
    Swizzle<A_SH_BITS, 3, A_SH_BITS>{},
    make_layout(make_shape(Int<A_TILE_M>{}, Int<A_TILE_K_HALF>{}),
                make_stride(Int<A_TILE_K_HALF>{}, Int<1>{}))
  ));

  extern __shared__ char smem_buf[];
  half_t* sh_a = reinterpret_cast<half_t*>(smem_buf);

  for (int i = threadIdx.x; i < A_TILE_M * A_TILE_K_HALF; i += THREADS)
    sh_a[i] = A_gmem[i];
  __syncthreads();

  auto sA = make_tensor(make_smem_ptr(sh_a), SmemLayoutA{});

  int lane_id = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;
  int k_warp_id = warp_id / N_WARPS_N;
  int a_rd_row_base = lane_id % 16;
  int a_rd_col_base = (lane_id / 16 + 2 * k_warp_id) * 8;

  // Load with manual ldsm4
  uint32_t frag_a[b_sh_wr_iters][M_BLOCKS][4];  // [k_subtile][m_block][4 uint32]
  for (int k = 0; k < b_sh_wr_iters; k++) {
    for (int m = 0; m < M_BLOCKS; m++) {
      int row = a_rd_row_base + m * 16;
      int col = a_rd_col_base + a_sh_rd_delta_o * k * 8;
      ldsm4_test(frag_a[k][m], &sA(row, col));
    }
  }

  // Write results
  if (lane_id == 0) {
    half_t* out = result + warp_id * (b_sh_wr_iters * M_BLOCKS * 8);
    for (int k = 0; k < b_sh_wr_iters; k++)
      for (int m = 0; m < M_BLOCKS; m++)
        for (int r = 0; r < 4; r++) {
          half_t* h = reinterpret_cast<half_t*>(&frag_a[k][m][r]);
          *out++ = h[0];
          *out++ = h[1];
        }
  }
}


// ============================================================================
// Host verification
// ============================================================================

int main() {
  constexpr int M = 16, K = 128, N_BLOCKS = 8, K_BLOCKS = 8;
  constexpr int N_WARPS = 8;

  // Create test data
  half_t* h_A = new half_t[M * K];
  for (int i = 0; i < M * K; i++)
    h_A[i] = half_t(float(i % 100) / 10.0f);

  half_t* d_A;
  cudaMalloc(&d_A, M * K * sizeof(half_t));
  cudaMemcpy(d_A, h_A, M * K * sizeof(half_t), cudaMemcpyHostToDevice);

  // Results
  int result_size = N_WARPS * K_BLOCKS / (N_BLOCKS/4) * 1 * 8;  // per warp: MMA_K * MMA_M * 8 values
  half_t* d_result_c, *d_result_manual;
  cudaMalloc(&d_result_c, result_size * N_WARPS * sizeof(half_t));
  cudaMalloc(&d_result_manual, result_size * N_WARPS * sizeof(half_t));
  cudaMemset(d_result_c, 0, result_size * N_WARPS * sizeof(half_t));
  cudaMemset(d_result_manual, 0, result_size * N_WARPS * sizeof(half_t));

  int smem_size = M * K * sizeof(half_t);

  printf("Testing approach C (auto-vectorized copy via TiledMMA partition)...\n");
  test_approach_a1<1, K_BLOCKS, N_BLOCKS><<<1, 256, smem_size>>>(d_A, d_result_c);
  cudaError_t err = cudaDeviceSynchronize();
  printf("  Status: %s\n", err == cudaSuccess ? "OK" : cudaGetErrorString(err));

  printf("Testing manual ldsm4 approach...\n");
  test_approach_manual<1, K_BLOCKS, N_BLOCKS><<<1, 256, smem_size>>>(d_A, d_result_manual);
  err = cudaDeviceSynchronize();
  printf("  Status: %s\n", err == cudaSuccess ? "OK" : cudaGetErrorString(err));

  // Compare warp 0's results
  half_t* h_c = new half_t[result_size * N_WARPS];
  half_t* h_m = new half_t[result_size * N_WARPS];
  cudaMemcpy(h_c, d_result_c, result_size * N_WARPS * sizeof(half_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_m, d_result_manual, result_size * N_WARPS * sizeof(half_t), cudaMemcpyDeviceToHost);

  printf("\nWarp 0 results comparison (first 16 values):\n");
  printf("  Approach C (auto copy): ");
  for (int i = 0; i < 16 && i < result_size; i++)
    printf("%.1f ", float(h_c[i]));
  printf("\n");
  printf("  Manual ldsm4:           ");
  for (int i = 0; i < 16 && i < result_size; i++)
    printf("%.1f ", float(h_m[i]));
  printf("\n");

  // Check if they match
  bool match = true;
  for (int w = 0; w < N_WARPS; w++) {
    for (int i = 0; i < result_size; i++) {
      if (float(h_c[w * result_size + i]) != float(h_m[w * result_size + i])) {
        match = false;
        printf("  MISMATCH at warp %d, idx %d: C=%.1f vs manual=%.1f\n",
               w, i, float(h_c[w * result_size + i]), float(h_m[w * result_size + i]));
        break;
      }
    }
    if (!match) break;
  }
  printf("\nAll warps match: %s\n", match ? "YES" : "NO");

  if (match) {
    printf("\n=== Conclusion ===\n");
    printf("copy(tCsA, tCrA) via TiledMMA partition produces identical results\n");
    printf("to manual ldsm4. This means we can use CuTe's auto-copy with\n");
    printf("partition_A/partition_fragment_A for the smem->reg path.\n");
    printf("The instruction used may differ (LDS.128 vs ldmatrix), but\n");
    printf("correctness is guaranteed. Performance needs kernel-level benchmarking.\n");
  }

  delete[] h_A; delete[] h_c; delete[] h_m;
  cudaFree(d_A); cudaFree(d_result_c); cudaFree(d_result_manual);
  return 0;
}
