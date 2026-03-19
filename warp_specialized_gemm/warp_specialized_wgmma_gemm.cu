/***************************************************************************************************
 * Cooperative Warp-Specialized WGMMA GEMM — CuTe Tutorial Style
 *
 * Demonstrates true warp specialization on Hopper (SM90):
 *   - Producer warp group (128 threads): TMA data loading from global → shared memory
 *   - Consumer warp group (128 threads): WGMMA matrix multiply-accumulate
 *
 * Key features:
 *   - PipelineTmaAsync for producer-consumer synchronization
 *   - Register reconfig (setmaxnreg) to optimize register allocation per role
 *   - GMMA SS mode: both operands sourced from shared memory via descriptors
 *   - Deep pipeline (7 stages) overlapping TMA loads with WGMMA compute
 *   - TMA descriptor prefetching for reduced first-load latency
 *   - GMMA ScaleOut::Zero for efficient accumulator initialization
 *   - CTA swizzle for L2 cache locality
 *   - Cluster TMA multicast: B tile shared across M-dimension cluster blocks
 *
 * Reference:
 *   - CUTLASS examples/cute/tutorial/hopper/wgmma_tma_sm90.cu (style)
 *   - CUTLASS include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp (logic)
 *   - CUTLASS include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp (kernel dispatch)
 *   - CUTLASS examples/48_hopper_warp_specialized_gemm (cluster + persistent reference)
 *
 * Build:  make
 * Run:    ./warp_specialized_wgmma_gemm
 *
 **************************************************************************************************/
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/arch/reg_reconfig.h"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

#include <cublas_v2.h>

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SharedStorage
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class ElementA,
          class ElementB,
          class SmemLayoutA,  // (M,K,PIPE)
          class SmemLayoutB,  // (N,K,PIPE)
          int   Stages>
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> smem_A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> smem_B;

  typename cutlass::PipelineTmaAsync<Stages>::SharedStorage pipeline;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Warp-Specialized GEMM Kernel with Cluster Multicast
//
// 256 threads total:
//   warp_group 0 (threads   0-127) = Producer (TMA loads)
//   warp_group 1 (threads 128-255) = Consumer (WGMMA compute)
//
// ClusterShape: (ClusterM, 1, 1)
//   B is multicast across ClusterM blocks sharing the same N-tile
//   A is loaded independently per block (no multicast)
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta,
          int Stages, int Log2Swizzle,
          class ClusterShape>
__global__ static
__launch_bounds__(256, 1)
void
gemm_device_warp_specialized(ProblemShape shape_MNK, CtaTiler cta_tiler,
                             TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                             TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                             TC      * C, CStride dC, TiledMma mma,
                             Alpha alpha, Beta beta)
{
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));  // BLK_K

  // Cluster dimensions
  constexpr int ClusterM = size<0>(ClusterShape{});
  constexpr int ClusterN = size<1>(ClusterShape{});

  // -----------------------------------------------------------------------
  // Setup shared memory
  // -----------------------------------------------------------------------

  extern __shared__ char shared_memory[];
  using SharedStorageT = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB, Stages>;
  SharedStorageT& smem = *reinterpret_cast<SharedStorageT*>(shared_memory);

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.begin()), SmemLayoutB{});

  // -----------------------------------------------------------------------
  // Full tensors → per-CTA tiles (with CTA swizzle + cluster awareness)
  // -----------------------------------------------------------------------

  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

  int m_blocks = (M + size<0>(cta_tiler) - 1) / size<0>(cta_tiler);
  int n_blocks = (N + size<1>(cta_tiler) - 1) / size<1>(cta_tiler);

  constexpr int SwizzleSize = (1 << Log2Swizzle);
  int linear_idx = int(blockIdx.x);
  // Map: group over SwizzleSize M-tiles, iterate N first within each group
  int group_n_tiles = SwizzleSize * n_blocks;
  int group_idx = linear_idx / group_n_tiles;
  int local_idx = linear_idx % group_n_tiles;
  int m_block = group_idx * SwizzleSize + local_idx % SwizzleSize;
  int n_block = local_idx / SwizzleSize;

  // With clusters, ALL blocks must participate in pipeline barriers and cluster_sync.
  // Out-of-bounds blocks clamp to valid coordinates, run the full pipeline (computing
  // on the same data as a valid partner), but skip the epilogue write to avoid corruption.
  bool is_valid_block = (m_block < m_blocks && n_block < n_blocks);
  m_block = min(m_block, m_blocks - 1);
  n_block = min(n_block, n_blocks - 1);

  auto cta_coord = make_coord(m_block, n_block, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1,  X>{}); // (BLK_M,BLK_N)

  // Always run the pipeline even for padded blocks (clamped coords give valid k_tile_count)
  int k_tile_count = size<2>(gA);

  // -----------------------------------------------------------------------
  // Cluster info for multicast
  // -----------------------------------------------------------------------

  uint32_t block_rank_in_cluster_ = cute::block_rank_in_cluster();

  // For ClusterShape (ClusterM, 1, 1):
  //   cluster_local_block_id.x = block_rank % ClusterM (position in M-dimension)
  //   cluster_local_block_id.y = block_rank / ClusterM (position in N-dimension)
  uint32_t cluster_local_m = block_rank_in_cluster_ % ClusterM;
  uint32_t cluster_local_n = block_rank_in_cluster_ / ClusterM;

  // -----------------------------------------------------------------------
  // Determine warp group role and pipeline setup
  // -----------------------------------------------------------------------

  int thread_idx = int(threadIdx.x);
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
  int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
  int lane_predicate = cute::elect_one_sync();

  bool is_producer_warp_group = (warp_group_idx == 0);

  // -----------------------------------------------------------------------
  // Prefetch TMA descriptors
  // -----------------------------------------------------------------------

  if ((warp_idx == 0) && lane_predicate) {
    cute::prefetch_tma_descriptor(tma_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(tma_b.get_tma_descriptor());
  }

  // -----------------------------------------------------------------------
  // Initialize PipelineTmaAsync
  // -----------------------------------------------------------------------

  using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
  using PipelineState    = cutlass::PipelineState<Stages>;

  constexpr int tma_transaction_bytes =
      sizeof(TA) * size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{})
    + sizeof(TB) * size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{});

  typename MainloopPipeline::Params pipeline_params;
  pipeline_params.transaction_bytes = tma_transaction_bytes;
  pipeline_params.num_consumers = cutlass::NumThreadsPerWarpGroup;

  if (is_producer_warp_group && warp_idx_in_warp_group == 0) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  } else if (!is_producer_warp_group) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  } else {
    pipeline_params.role = MainloopPipeline::ThreadCategory::NonParticipant;
  }
  pipeline_params.is_leader = warp_group_thread_idx == 0;

  // Pass actual cluster shape for barrier initialization
  MainloopPipeline pipeline(smem.pipeline, pipeline_params, ClusterShape{});

  // Cluster-wide sync to ensure all barriers are initialized
  if constexpr (size(ClusterShape{}) > 1) {
    cute::cluster_sync();
  } else {
    __syncthreads();
  }

  // -----------------------------------------------------------------------
  // Branch into Producer vs Consumer
  // -----------------------------------------------------------------------

  if (is_producer_warp_group) {
    // =================================================================
    // PRODUCER: TMA data loading with multicast
    // =================================================================

    cutlass::arch::warpgroup_reg_dealloc<40>();

    if (lane_predicate && warp_idx_in_warp_group == 0) {

      // TMA partitioning
      // A: no multicast (each block loads its own A tile)
      auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                        group_modes<0,2>(sA), group_modes<0,2>(gA));

      // B: multicast across ClusterM blocks sharing the same N-tile
      auto [tBgB, tBsB] = tma_partition(tma_b, cluster_local_m,
                                        make_layout(Int<ClusterM>{}),
                                        group_modes<0,2>(sB), group_modes<0,2>(gB));

      // Compute multicast mask for B
      uint16_t mcast_mask_b = 0;
      auto cluster_layout = Layout<ClusterShape>{};
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < ClusterM; ++m) {
        mcast_mask_b |= (uint16_t(1) << cluster_layout(m, cluster_local_n, Int<0>{}));
      }

      PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();

      int k_tile = 0;
      CUTE_NO_UNROLL
      for (int count = k_tile_count; count > 0; --count) {
        pipeline.producer_acquire(smem_pipe_write);

        int pipe = smem_pipe_write.index();
        auto* tma_bar = pipeline.producer_get_barrier(smem_pipe_write);

        // A: no multicast
        copy(tma_a.with(*tma_bar), tAgA(_, k_tile), tAsA(_, pipe));
        // B: multicast across ClusterM blocks
        copy(tma_b.with(*tma_bar, mcast_mask_b), tBgB(_, k_tile), tBsB(_, pipe));

        ++smem_pipe_write;
        ++k_tile;
      }

      pipeline.producer_tail(smem_pipe_write);
    }

  } else {
    // =================================================================
    // CONSUMER: WGMMA computation
    // =================================================================

    cutlass::arch::warpgroup_reg_alloc<232>();

    {
      ThrMMA thr_mma = mma.get_thread_slice(warp_group_thread_idx);
      Tensor tCsA = thr_mma.partition_A(sA);
      Tensor tCsB = thr_mma.partition_B(sB);
      Tensor tCgC = thr_mma.partition_C(gC);

      Tensor tCrC = thr_mma.make_fragment_C(tCgC);
      Tensor tCrA = thr_mma.make_fragment_A(tCsA);
      Tensor tCrB = thr_mma.make_fragment_B(tCsB);

      PipelineState smem_pipe_read;
      PipelineState smem_pipe_release = smem_pipe_read;

      constexpr int K_PIPE_MMAS = 1;

      // Prologue: first k-tile with ScaleOut::Zero
      {
        auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
        pipeline.consumer_wait(smem_pipe_read, barrier_token);

        int read_stage = smem_pipe_read.index();

        mma.accumulate_ = GMMA::ScaleOut::Zero;

        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
          cute::gemm(mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), tCrC);
          mma.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        ++smem_pipe_read;
      }

      warpgroup_fence_operand(tCrC);

      // Main loop
      CUTE_NO_UNROLL
      for (int k_tile = 1; k_tile < k_tile_count; ++k_tile) {
        auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
        pipeline.consumer_wait(smem_pipe_read, barrier_token);

        int read_stage = smem_pipe_read.index();

        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        cute::gemm(mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), tCrC);
        warpgroup_commit_batch();

        warpgroup_wait<K_PIPE_MMAS>();
        warpgroup_fence_operand(tCrC);

        pipeline.consumer_release(smem_pipe_release);

        ++smem_pipe_read;
        ++smem_pipe_release;
      }

      // MMA tail
      warpgroup_wait<0>();

      for (int i = 0; i < K_PIPE_MMAS; ++i) {
        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_release;
      }

      // Epilogue
      if (is_valid_block) {
        axpby(alpha, tCrC, beta, tCgC);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host-side launch for NT layout
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);

  constexpr int Stages = 7;
  constexpr int Log2Swizzle = 3;
  constexpr int SwizzleSize = (1 << Log2Swizzle);

  // Cluster: 2 blocks along M share B tiles via TMA multicast
  constexpr int ClusterM = 2;
  constexpr int ClusterN = 1;
  using ClusterShape = Shape<Int<ClusterM>, Int<ClusterN>, _1>;

  auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM, bK, Int<Stages>{}));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN, bK, Int<Stages>{}));

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});

  Tensor mA = make_tensor(A, make_shape(M, K), dA);
  Tensor mB = make_tensor(B, make_shape(N, K), dB);

  // A: regular TMA load (no multicast)
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM, bK));
  // B: TMA multicast across ClusterM blocks
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD_MULTICAST{}, mB, sB(_,_,0),
                                 make_shape(bN, bK), Int<ClusterM>{});

  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB), Stages>));

  // 1D grid with swizzle, must be multiple of ClusterM
  int m_blocks = size(ceil_div(m, bM));
  int n_blocks = size(ceil_div(n, bN));
  int m_swizzle_groups = (m_blocks + SwizzleSize - 1) / SwizzleSize;
  int total_blocks = m_swizzle_groups * SwizzleSize * n_blocks;
  // Round up to multiple of cluster size
  total_blocks = ((total_blocks + ClusterM - 1) / ClusterM) * ClusterM;

  dim3 dimBlock(256);
  dim3 dimCluster(ClusterM, ClusterN, 1);
  dim3 dimGrid(total_blocks, 1, 1);

  auto* kernel_ptr = &gemm_device_warp_specialized<
      decltype(prob_shape), decltype(cta_tiler),
      TA, decltype(sA), decltype(tmaA),
      TB, decltype(sB), decltype(tmaB),
      TC, decltype(dC), decltype(tiled_mma),
      decltype(alpha), decltype(beta),
      Stages, Log2Swizzle, ClusterShape>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, reinterpret_cast<void const*>(kernel_ptr),
      prob_shape, cta_tiler,
      A, tmaA,
      B, tmaB,
      C, dC, tiled_mma,
      alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "Error: kernel launch failed\n");
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host-side launch for TN layout
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(Int<1>{}, ldC);

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);

  constexpr int Stages = 7;
  constexpr int Log2Swizzle = 3;
  constexpr int SwizzleSize = (1 << Log2Swizzle);

  constexpr int ClusterM = 2;
  constexpr int ClusterN = 1;
  using ClusterShape = Shape<Int<ClusterM>, Int<ClusterN>, _1>;

  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, Int<Stages>{}));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK, Int<Stages>{}));

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});

  Tensor mA = make_tensor(A, make_shape(M, K), dA);
  Tensor mB = make_tensor(B, make_shape(N, K), dB);

  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM, bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD_MULTICAST{}, mB, sB(_,_,0),
                                 make_shape(bN, bK), Int<ClusterM>{});

  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB), Stages>));

  int m_blocks = size(ceil_div(m, bM));
  int n_blocks = size(ceil_div(n, bN));
  int m_swizzle_groups = (m_blocks + SwizzleSize - 1) / SwizzleSize;
  int total_blocks = m_swizzle_groups * SwizzleSize * n_blocks;
  total_blocks = ((total_blocks + ClusterM - 1) / ClusterM) * ClusterM;

  dim3 dimBlock(256);
  dim3 dimCluster(ClusterM, ClusterN, 1);
  dim3 dimGrid(total_blocks, 1, 1);

  auto* kernel_ptr = &gemm_device_warp_specialized<
      decltype(prob_shape), decltype(cta_tiler),
      TA, decltype(sA), decltype(tmaA),
      TB, decltype(sB), decltype(tmaB),
      TC, decltype(dC), decltype(tiled_mma),
      decltype(alpha), decltype(beta),
      Stages, Log2Swizzle, ClusterShape>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, reinterpret_cast<void const*>(kernel_ptr),
      prob_shape, cta_tiler,
      A, tmaA,
      B, tmaB,
      C, dC, tiled_mma,
      alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "Error: kernel launch failed\n");
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     TA const* A, int ldA,
     TB const* B, int ldB,
     Beta beta,
     TC      * C, int ldC,
     cudaStream_t stream = 0)
{
  if (transA == 'N' && transB == 'T') {
    return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  } else if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  }
  assert(false && "Only NT and TN layouts are implemented");
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// cuBLAS reference GEMM
//
///////////////////////////////////////////////////////////////////////////////////////////////////

void cublas_gemm(cublasHandle_t handle,
                 int m, int n, int k,
                 cute::half_t const* A, int ldA,
                 cute::half_t const* B, int ldB,
                 cute::half_t      * C, int ldC,
                 char transA, char transB)
{
  __half alpha_h = __float2half(1.0f);
  __half beta_h  = __float2half(0.0f);

  cublasOperation_t opA = (transA == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t opB = (transB == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;

  int lda_cublas = (transA == 'N') ? m : k;
  int ldb_cublas = (transB == 'N') ? k : n;

  cublasStatus_t stat = cublasHgemm(handle,
      opA, opB,
      m, n, k,
      &alpha_h,
      reinterpret_cast<__half const*>(A), lda_cublas,
      reinterpret_cast<__half const*>(B), ldb_cublas,
      &beta_h,
      reinterpret_cast<__half*>(C), m);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS HGEMM failed with status %d\n", stat);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

double max_relative_error(thrust::host_vector<cute::half_t> const& result,
                          thrust::host_vector<cute::half_t> const& reference,
                          int size)
{
  double max_err = 0.0;
  for (int i = 0; i < size; ++i) {
    double r = static_cast<double>(static_cast<float>(result[i]));
    double ref = static_cast<double>(static_cast<float>(reference[i]));
    double err = std::abs(r - ref) / std::max(std::abs(ref), 1e-5);
    max_err = std::max(max_err, err);
  }
  return max_err;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

bool run_correctness_test(cublasHandle_t handle, int m, int n, int k,
                          char transA, char transB)
{
  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = cute::half_t;

  thrust::host_vector<TA> h_A(m * k);
  thrust::host_vector<TB> h_B(n * k);
  thrust::host_vector<TC> h_C(m * n, TC(0));

  srand(42);
  for (int j = 0; j < m * k; ++j) h_A[j] = TA(float((rand() % 5) - 2));
  for (int j = 0; j < n * k; ++j) h_B[j] = TB(float((rand() % 5) - 2));

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C_cute = h_C;
  thrust::device_vector<TC> d_C_cublas = h_C;

  int ldA = (transA == 'N') ? m : k;
  int ldB = (transB == 'N') ? k : n;
  int ldC = m;

  TA alpha = TA(1.0f);
  TA beta  = TA(0.0f);

  cublas_gemm(handle, m, n, k,
              d_A.data().get(), ldA,
              d_B.data().get(), ldB,
              d_C_cublas.data().get(), ldC,
              transA, transB);
  cudaDeviceSynchronize();

  gemm(transA, transB, m, n, k,
       alpha,
       d_A.data().get(), ldA,
       d_B.data().get(), ldB,
       beta,
       d_C_cute.data().get(), ldC);
  CUTE_CHECK_LAST();
  cudaDeviceSynchronize();

  thrust::host_vector<TC> h_cute_result = d_C_cute;
  thrust::host_vector<TC> h_cublas_result = d_C_cublas;

  double err = max_relative_error(h_cute_result, h_cublas_result, m * n);
  double tolerance = 1e-2;

  bool passed = (err < tolerance);
  printf("  [%c%c] M=%4d N=%4d K=%4d  max_rel_err=%.6e  %s\n",
         transA, transB, m, n, k, err, passed ? "PASS" : "FAIL");

  return passed;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void run_benchmark(cublasHandle_t handle, int m, int n, int k,
                   char transA, char transB)
{
  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = cute::half_t;

  thrust::host_vector<TA> h_A(m * k);
  thrust::host_vector<TB> h_B(n * k);
  thrust::host_vector<TC> h_C(m * n, TC(0));

  for (int j = 0; j < m * k; ++j) h_A[j] = TA(float((rand() % 5) - 2));
  for (int j = 0; j < n * k; ++j) h_B[j] = TB(float((rand() % 5) - 2));

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  int ldA = (transA == 'N') ? m : k;
  int ldB = (transB == 'N') ? k : n;
  int ldC = m;

  TA alpha = TA(1.0f);
  TA beta  = TA(0.0f);

  double gflops = (2.0 * m * n * k) * 1e-9;

  const int warmup = 10;
  const int timing = 100;

  GPU_Clock timer;

  for (int i = 0; i < warmup; ++i) {
    gemm(transA, transB, m, n, k, alpha,
         d_A.data().get(), ldA, d_B.data().get(), ldB,
         beta, d_C.data().get(), ldC);
  }
  cudaDeviceSynchronize();

  timer.start();
  for (int i = 0; i < timing; ++i) {
    gemm(transA, transB, m, n, k, alpha,
         d_A.data().get(), ldA, d_B.data().get(), ldB,
         beta, d_C.data().get(), ldC);
  }
  double cute_time = timer.seconds() / timing;
  CUTE_CHECK_LAST();

  __half alpha_h = __float2half(1.0f);
  __half beta_h  = __float2half(0.0f);
  cublasOperation_t opA = (transA == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t opB = (transB == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  int lda_cublas = (transA == 'N') ? m : k;
  int ldb_cublas = (transB == 'N') ? k : n;

  for (int i = 0; i < warmup; ++i) {
    cublasHgemm(handle, opA, opB, m, n, k,
                &alpha_h,
                reinterpret_cast<__half const*>(d_A.data().get()), lda_cublas,
                reinterpret_cast<__half const*>(d_B.data().get()), ldb_cublas,
                &beta_h,
                reinterpret_cast<__half*>(d_C.data().get()), m);
  }
  cudaDeviceSynchronize();

  timer.start();
  for (int i = 0; i < timing; ++i) {
    cublasHgemm(handle, opA, opB, m, n, k,
                &alpha_h,
                reinterpret_cast<__half const*>(d_A.data().get()), lda_cublas,
                reinterpret_cast<__half const*>(d_B.data().get()), ldb_cublas,
                &beta_h,
                reinterpret_cast<__half*>(d_C.data().get()), m);
  }
  double cublas_time = timer.seconds() / timing;

  double cute_gflops   = gflops / cute_time;
  double cublas_gflops  = gflops / cublas_time;
  double efficiency_pct = cute_gflops / cublas_gflops * 100.0;

  printf("  [%c%c] M=%4d N=%4d K=%4d  | CuTe: %7.1f GFlop/s (%6.4f ms) | cuBLAS: %7.1f GFlop/s (%6.4f ms) | %.1f%%\n",
         transA, transB, m, n, k,
         cute_gflops, cute_time * 1000,
         cublas_gflops, cublas_time * 1000,
         efficiency_pct);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  cudaDeviceProp props;
  int device_id;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&props, device_id);

  if (props.major != 9) {
    printf("This example requires Hopper (SM90) GPU. Found SM%d%d.\n",
           props.major, props.minor);
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  printf("=== Cooperative Warp-Specialized WGMMA GEMM (Cluster Multicast) ===\n");
  printf("GPU: %s (SM%d%d)\n\n", props.name, props.major, props.minor);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

  printf("--- Correctness Tests (vs cuBLAS) ---\n");
  bool all_passed = true;

  struct TestSize { int m, n, k; };
  std::vector<TestSize> test_sizes = {
    {128,  128,   64},
    {256,  256,  256},
    {512,  512,  512},
    {1024, 2048, 512},
    {2048, 2048, 2048},
    {4096, 4096, 4096},
  };

  for (auto& s : test_sizes) {
    all_passed &= run_correctness_test(handle, s.m, s.n, s.k, 'N', 'T');
  }

  printf("\n  TN layout:\n");
  for (auto& s : test_sizes) {
    all_passed &= run_correctness_test(handle, s.m, s.n, s.k, 'T', 'N');
  }

  printf("\nOverall: %s\n\n", all_passed ? "ALL PASSED" : "SOME FAILED");

  printf("--- Performance Benchmarks (100 iterations) ---\n");

  std::vector<TestSize> bench_sizes = {
    {1024, 1024, 1024},
    {2048, 2048, 2048},
    {4096, 4096, 4096},
    {8192, 8192, 8192},
  };

  printf("\n  NT layout:\n");
  for (auto& s : bench_sizes) {
    run_benchmark(handle, s.m, s.n, s.k, 'N', 'T');
  }

  printf("\n  TN layout:\n");
  for (auto& s : bench_sizes) {
    run_benchmark(handle, s.m, s.n, s.k, 'T', 'N');
  }

  cublasDestroy(handle);

#else
  printf("CUTLASS_ARCH_MMA_SM90_SUPPORTED must be enabled. Test waived.\n");
#endif

  return 0;
}
