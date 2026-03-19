/***************************************************************************************************
 * Pingpong Warp-Specialized WGMMA GEMM — CuTe Tutorial Style
 *
 * Two consumer warp groups alternate processing DIFFERENT tiles:
 *   - Consumer0: tiles 0, 2, 4, ...
 *   - Consumer1: tiles 1, 3, 5, ...
 * Producer continuously loads K-blocks for both tiles without draining.
 *
 * 384 threads total:
 *   WG 0 (threads   0-127) = Producer (TMA loads, continuous across tiles)
 *   WG 1 (threads 128-255) = Consumer0 (WGMMA, processes even-indexed tiles)
 *   WG 2 (threads 256-383) = Consumer1 (WGMMA, processes odd-indexed tiles)
 *
 * Key advantage over basic persistent: NO producer_tail drain between tiles!
 * The pipeline flows continuously — while Consumer0 writes epilogue for tile N,
 * Consumer1 is already computing tile N+1.
 *
 * Build:  make pingpong
 * Run:    ./warp_specialized_pingpong_gemm
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

template <class ElementA, class ElementB,
          class SmemLayoutA, class SmemLayoutB, int Stages>
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> smem_A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> smem_B;
  typename cutlass::PipelineTmaAsync<Stages>::SharedStorage pipeline;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Pingpong Warp-Specialized GEMM Kernel
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta, int Stages, int Log2Swizzle>
__global__ static
__launch_bounds__(384, 1)
void
gemm_device_pingpong(ProblemShape shape_MNK, CtaTiler cta_tiler,
                     TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                     TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                     TC      * C, CStride dC, TiledMma mma,
                     Alpha alpha, Beta beta, int total_tiles)
{
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  extern __shared__ char shared_memory[];
  using SharedStorageT = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB, Stages>;
  SharedStorageT& smem = *reinterpret_cast<SharedStorageT*>(shared_memory);

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.begin()), SmemLayoutB{});

  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

  int m_blocks = (M + size<0>(cta_tiler) - 1) / size<0>(cta_tiler);
  int n_blocks = (N + size<1>(cta_tiler) - 1) / size<1>(cta_tiler);
  int k_tile_count = (K + size<2>(cta_tiler) - 1) / size<2>(cta_tiler);

  constexpr int SwizzleSize = (1 << Log2Swizzle);

  // Swizzle: map linear tile index → (m_block, n_block)
  auto tile_to_coord = [&](int linear_idx) -> cute::tuple<int, int, bool> {
    int group_n_tiles = SwizzleSize * n_blocks;
    int gidx = linear_idx / group_n_tiles;
    int lidx = linear_idx % group_n_tiles;
    int mb = gidx * SwizzleSize + lidx % SwizzleSize;
    int nb = lidx / SwizzleSize;
    bool valid = (mb < m_blocks && nb < n_blocks);
    return {min(mb, m_blocks - 1), min(nb, n_blocks - 1), valid};
  };

  // -----------------------------------------------------------------------
  // Warp group roles
  // -----------------------------------------------------------------------

  int thread_idx = int(threadIdx.x);
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
  int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
  int lane_predicate = cute::elect_one_sync();

  // WG 0 = Producer, WG 1 = Consumer0, WG 2 = Consumer1
  bool is_producer = (warp_group_idx == 0);
  int consumer_idx = warp_group_idx - 1;  // 0 or 1 for consumers

  // -----------------------------------------------------------------------
  // Prefetch TMA descriptors
  // -----------------------------------------------------------------------

  if ((warp_idx == 0) && lane_predicate) {
    cute::prefetch_tma_descriptor(tma_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(tma_b.get_tma_descriptor());
  }

  // -----------------------------------------------------------------------
  // Pipeline: 2 consumer WGs = 256 consumer threads
  // -----------------------------------------------------------------------

  using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
  using PipelineState    = cutlass::PipelineState<Stages>;

  constexpr int tma_transaction_bytes =
      sizeof(TA) * size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{})
    + sizeof(TB) * size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{});

  typename MainloopPipeline::Params pipeline_params;
  pipeline_params.transaction_bytes = tma_transaction_bytes;
  // Each consumer WG independently releases its stages (128 threads per release)
  pipeline_params.num_consumers = cutlass::NumThreadsPerWarpGroup;

  if (is_producer && warp_idx_in_warp_group == 0) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  } else if (!is_producer) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  } else {
    pipeline_params.role = MainloopPipeline::ThreadCategory::NonParticipant;
  }
  pipeline_params.is_leader = warp_group_thread_idx == 0;

  MainloopPipeline pipeline(smem.pipeline, pipeline_params, Shape<_1, _1, _1>{});

  __syncthreads();

  // -----------------------------------------------------------------------
  // Each CTA processes tile pairs: (2*blockIdx.x, 2*blockIdx.x+1),
  //   then (2*(blockIdx.x+gridDim.x), 2*(blockIdx.x+gridDim.x)+1), etc.
  // Producer loads tiles continuously. Consumer0 does even, Consumer1 does odd.
  // -----------------------------------------------------------------------

  // Each CTA handles exactly 1 tile pair (non-persistent for now)
  int pair_idx = int(blockIdx.x);
  int tile0_idx = pair_idx * 2;          // Consumer0's tile
  int tile1_idx = pair_idx * 2 + 1;     // Consumer1's tile
  bool has_tile1 = (tile1_idx < total_tiles);

  if (is_producer) {
    // =================================================================
    // PRODUCER: load tile0 then tile1 continuously (no drain between)
    // =================================================================

    cutlass::arch::warpgroup_reg_dealloc<40>();

    if (lane_predicate && warp_idx_in_warp_group == 0) {
      PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();

      // Load tile0
      {
        auto [mb, nb, valid] = tile_to_coord(tile0_idx);
        Tensor gA_t = local_tile(mA, cta_tiler, make_coord(mb, nb, _), Step<_1, X, _1>{});
        Tensor gB_t = local_tile(mB, cta_tiler, make_coord(mb, nb, _), Step< X, _1, _1>{});
        auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                          group_modes<0,2>(sA), group_modes<0,2>(gA_t));
        auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                          group_modes<0,2>(sB), group_modes<0,2>(gB_t));
        for (int k = 0; k < k_tile_count; ++k) {
          pipeline.producer_acquire(smem_pipe_write);
          auto* bar = pipeline.producer_get_barrier(smem_pipe_write);
          copy(tma_a.with(*bar), tAgA(_, k), tAsA(_, smem_pipe_write.index()));
          copy(tma_b.with(*bar), tBgB(_, k), tBsB(_, smem_pipe_write.index()));
          ++smem_pipe_write;
        }
      }

      // Load tile1 — DISABLED for debug
      if (false && has_tile1) {
        auto [mb, nb, valid] = tile_to_coord(tile1_idx);
        Tensor gA_t = local_tile(mA, cta_tiler, make_coord(mb, nb, _), Step<_1, X, _1>{});
        Tensor gB_t = local_tile(mB, cta_tiler, make_coord(mb, nb, _), Step< X, _1, _1>{});
        auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                          group_modes<0,2>(sA), group_modes<0,2>(gA_t));
        auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                          group_modes<0,2>(sB), group_modes<0,2>(gB_t));
        for (int k = 0; k < k_tile_count; ++k) {
          pipeline.producer_acquire(smem_pipe_write);
          auto* bar = pipeline.producer_get_barrier(smem_pipe_write);
          copy(tma_a.with(*bar), tAgA(_, k), tAsA(_, smem_pipe_write.index()));
          copy(tma_b.with(*bar), tBgB(_, k), tBsB(_, smem_pipe_write.index()));
          ++smem_pipe_write;
        }
      }

      pipeline.producer_tail(smem_pipe_write);
    }

  } else {
    // =================================================================
    // CONSUMER0 (consumer_idx=0): processes tile0
    // CONSUMER1 (consumer_idx=1): processes tile1
    // =================================================================

    cutlass::arch::warpgroup_reg_alloc<232>();

    // Consumer1 skips past tile0's stages
    PipelineState smem_pipe_read;
    if (consumer_idx == 1) {
      // Manually advance by k_tile_count, handling wraps correctly
      for (int i = 0; i < k_tile_count; ++i) { ++smem_pipe_read; }
    }
    PipelineState smem_pipe_release = smem_pipe_read;

    int my_tile = tile0_idx;  // DEBUG: both consumers process same tile
    bool should_run = (consumer_idx == 0);  // DEBUG: only Consumer0 runs

    if (should_run) {
      ThrMMA thr_mma = mma.get_thread_slice(warp_group_thread_idx);
      Tensor tCsA = thr_mma.partition_A(sA);
      Tensor tCsB = thr_mma.partition_B(sB);
      Tensor tCrA = thr_mma.make_fragment_A(tCsA);
      Tensor tCrB = thr_mma.make_fragment_B(tCsB);

      auto [mb, nb, valid] = tile_to_coord(my_tile);
      Tensor gC_tile = local_tile(mC, cta_tiler, make_coord(mb, nb, _), Step<_1, _1, X>{});
      Tensor tCgC = thr_mma.partition_C(gC_tile);
      Tensor tCrC = thr_mma.make_fragment_C(tCgC);

      constexpr int K_PIPE_MMAS = 1;

      // Prologue
      {
        auto bt = pipeline.consumer_try_wait(smem_pipe_read);
        pipeline.consumer_wait(smem_pipe_read, bt);
        int rs = smem_pipe_read.index();
        mma.accumulate_ = GMMA::ScaleOut::Zero;
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        CUTLASS_PRAGMA_UNROLL
        for (int kb = 0; kb < size<2>(tCrA); ++kb) {
          cute::gemm(mma, tCrA(_,_,kb,rs), tCrB(_,_,kb,rs), tCrC);
          mma.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        ++smem_pipe_read;
      }
      warpgroup_fence_operand(tCrC);

      // Main loop
      CUTE_NO_UNROLL
      for (int k = 1; k < k_tile_count; ++k) {
        auto bt = pipeline.consumer_try_wait(smem_pipe_read);
        pipeline.consumer_wait(smem_pipe_read, bt);
        int rs = smem_pipe_read.index();
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        cute::gemm(mma, tCrA(_,_,_,rs), tCrB(_,_,_,rs), tCrC);
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
      if (valid) { axpby(alpha, tCrC, beta, tCgC); }

    } else {
      // Consumer1 with no tile1: must still release k_tile_count stages
      // so producer_tail doesn't deadlock. But there are no stages to release
      // since producer didn't load anything for tile1.
      // Nothing to do — producer_tail only waits on stages it actually wrote.
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host launch for NT layout
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class TA, class TB, class TC, class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha, TA const* A, int ldA, TB const* B, int ldB,
        Beta beta, TC* C, int ldC, cudaStream_t stream = 0)
{
  auto M = int(m), N = int(n), K = int(k);
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

  auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM, bK, Int<Stages>{}));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN, bK, Int<Stages>{}));

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});

  Tensor mA = make_tensor(A, make_shape(M, K), dA);
  Tensor mB = make_tensor(B, make_shape(N, K), dB);
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM, bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN, bK));

  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB), Stages>));

  int m_blocks = size(ceil_div(m, bM));
  int n_blocks = size(ceil_div(n, bN));
  int m_swizzle_groups = (m_blocks + SwizzleSize - 1) / SwizzleSize;
  int total_tiles = m_swizzle_groups * SwizzleSize * n_blocks;

  // Non-persistent: 1 CTA per tile pair
  int total_tile_pairs = (total_tiles + 1) / 2;
  int grid_blocks = total_tile_pairs;

  dim3 dimBlock(384);
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(grid_blocks, 1, 1);

  auto* kernel_ptr = &gemm_device_pingpong<
      decltype(prob_shape), decltype(cta_tiler),
      TA, decltype(sA), decltype(tmaA),
      TB, decltype(sB), decltype(tmaB),
      TC, decltype(dC), decltype(tiled_mma),
      decltype(alpha), decltype(beta),
      Stages, Log2Swizzle>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, reinterpret_cast<void const*>(kernel_ptr),
      prob_shape, cta_tiler,
      A, tmaA, B, tmaB,
      C, dC, tiled_mma, alpha, beta, total_tiles);
  CUTE_CHECK_LAST();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host launch for TN layout
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class TA, class TB, class TC, class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha, TA const* A, int ldA, TB const* B, int ldB,
        Beta beta, TC* C, int ldC, cudaStream_t stream = 0)
{
  auto M = int(m), N = int(n), K = int(k);
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

  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, Int<Stages>{}));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK, Int<Stages>{}));

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});

  Tensor mA = make_tensor(A, make_shape(M, K), dA);
  Tensor mB = make_tensor(B, make_shape(N, K), dB);
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM, bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN, bK));

  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB), Stages>));

  int m_blocks = size(ceil_div(m, bM));
  int n_blocks = size(ceil_div(n, bN));
  int m_swizzle_groups = (m_blocks + SwizzleSize - 1) / SwizzleSize;
  int total_tiles = m_swizzle_groups * SwizzleSize * n_blocks;

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  int total_tile_pairs = (total_tiles + 1) / 2;
  int grid_blocks = min(total_tile_pairs, props.multiProcessorCount);

  dim3 dimBlock(384);
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(grid_blocks, 1, 1);

  auto* kernel_ptr = &gemm_device_pingpong<
      decltype(prob_shape), decltype(cta_tiler),
      TA, decltype(sA), decltype(tmaA),
      TB, decltype(sB), decltype(tmaB),
      TC, decltype(dC), decltype(tiled_mma),
      decltype(alpha), decltype(beta),
      Stages, Log2Swizzle>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, reinterpret_cast<void const*>(kernel_ptr),
      prob_shape, cta_tiler,
      A, tmaA, B, tmaB,
      C, dC, tiled_mma, alpha, beta, total_tiles);
  CUTE_CHECK_LAST();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(char transA, char transB, int m, int n, int k,
          Alpha alpha, TA const* A, int ldA, TB const* B, int ldB,
          Beta beta, TC* C, int ldC, cudaStream_t stream = 0)
{
  if (transA == 'N' && transB == 'T')
    return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  else if (transA == 'T' && transB == 'N')
    return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  assert(false && "Only NT and TN layouts implemented");
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void cublas_gemm(cublasHandle_t handle, int m, int n, int k,
                 cute::half_t const* A, cute::half_t const* B, cute::half_t* C,
                 char transA, char transB)
{
  __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
  cublasOperation_t opA = (transA == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t opB = (transB == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  int lda = (transA == 'N') ? m : k, ldb = (transB == 'N') ? k : n;
  cublasHgemm(handle, opA, opB, m, n, k, &alpha_h,
              reinterpret_cast<__half const*>(A), lda,
              reinterpret_cast<__half const*>(B), ldb,
              &beta_h, reinterpret_cast<__half*>(C), m);
}

double max_relative_error(thrust::host_vector<cute::half_t> const& r,
                          thrust::host_vector<cute::half_t> const& ref, int sz)
{
  double mx = 0.0;
  for (int i = 0; i < sz; ++i) {
    double a = float(r[i]), b = float(ref[i]);
    mx = std::max(mx, std::abs(a-b) / std::max(std::abs(b), 1e-5));
  }
  return mx;
}

bool run_test(cublasHandle_t h, int m, int n, int k, char tA, char tB) {
  using T = cute::half_t;
  thrust::host_vector<T> hA(m*k), hB(n*k), hC(m*n, T(0));
  srand(42);
  for (auto& v : hA) v = T(float((rand()%5)-2));
  for (auto& v : hB) v = T(float((rand()%5)-2));
  thrust::device_vector<T> dA=hA, dB=hB, dC1=hC, dC2=hC;
  int ldA=(tA=='N')?m:k, ldB=(tB=='N')?k:n;
  cublas_gemm(h,m,n,k,dA.data().get(),dB.data().get(),dC1.data().get(),tA,tB);
  cudaDeviceSynchronize();
  gemm(tA,tB,m,n,k,T(1.f),dA.data().get(),ldA,dB.data().get(),ldB,T(0.f),dC2.data().get(),m);
  CUTE_CHECK_LAST(); cudaDeviceSynchronize();
  double err = max_relative_error(thrust::host_vector<T>(dC2), thrust::host_vector<T>(dC1), m*n);
  bool ok = err < 1e-2;
  printf("  [%c%c] M=%4d N=%4d K=%4d  err=%.2e  %s\n", tA,tB,m,n,k,err,ok?"PASS":"FAIL");
  return ok;
}

void run_bench(cublasHandle_t h, int m, int n, int k, char tA, char tB) {
  using T = cute::half_t;
  thrust::host_vector<T> hA(m*k), hB(n*k), hC(m*n, T(0));
  for (auto& v : hA) v = T(float((rand()%5)-2));
  for (auto& v : hB) v = T(float((rand()%5)-2));
  thrust::device_vector<T> dA=hA, dB=hB, dC=hC;
  int ldA=(tA=='N')?m:k, ldB=(tB=='N')?k:n;
  double gf = 2.0*m*n*k*1e-9;
  GPU_Clock timer;
  for (int i=0;i<10;i++) gemm(tA,tB,m,n,k,T(1.f),dA.data().get(),ldA,dB.data().get(),ldB,T(0.f),dC.data().get(),m);
  cudaDeviceSynchronize();
  timer.start();
  for (int i=0;i<100;i++) gemm(tA,tB,m,n,k,T(1.f),dA.data().get(),ldA,dB.data().get(),ldB,T(0.f),dC.data().get(),m);
  double ct = timer.seconds()/100; CUTE_CHECK_LAST();
  __half a16=__float2half(1.f),b16=__float2half(0.f);
  auto opA=(tA=='N')?CUBLAS_OP_N:CUBLAS_OP_T, opB=(tB=='N')?CUBLAS_OP_N:CUBLAS_OP_T;
  int la=(tA=='N')?m:k, lb=(tB=='N')?k:n;
  for(int i=0;i<10;i++) cublasHgemm(h,opA,opB,m,n,k,&a16,(__half*)dA.data().get(),la,(__half*)dB.data().get(),lb,&b16,(__half*)dC.data().get(),m);
  cudaDeviceSynchronize();
  timer.start();
  for(int i=0;i<100;i++) cublasHgemm(h,opA,opB,m,n,k,&a16,(__half*)dA.data().get(),la,(__half*)dB.data().get(),lb,&b16,(__half*)dC.data().get(),m);
  double bt=timer.seconds()/100;
  printf("  [%c%c] M=%4d N=%4d K=%4d  | PP: %7.1f GF/s (%.4fms) | cuBLAS: %7.1f GF/s (%.4fms) | %.1f%%\n",
         tA,tB,m,n,k, gf/ct,ct*1e3, gf/bt,bt*1e3, (gf/ct)/(gf/bt)*100);
}

int main(int argc, char** argv) {
  cudaDeviceProp props; int dev; cudaGetDevice(&dev); cudaGetDeviceProperties(&props, dev);
  if (props.major != 9) { printf("Need SM90\n"); return 0; }
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  printf("=== Pingpong Warp-Specialized WGMMA GEMM ===\n");
  printf("GPU: %s\n\n", props.name);
  cublasHandle_t h; cublasCreate(&h);
  printf("--- Correctness ---\n");
  bool ok = true;
  struct S { int m,n,k; };
  for (auto s : std::vector<S>{{128,128,64},{256,256,256},{512,512,512},{1024,2048,512},{2048,2048,2048},{4096,4096,4096}}) {
    ok &= run_test(h,s.m,s.n,s.k,'N','T');
    ok &= run_test(h,s.m,s.n,s.k,'T','N');
  }
  printf("\nOverall: %s\n\n", ok?"ALL PASSED":"SOME FAILED");
  printf("--- Performance (100 iters) ---\n");
  for (auto s : std::vector<S>{{1024,1024,1024},{2048,2048,2048},{4096,4096,4096},{8192,8192,8192}}) {
    run_bench(h,s.m,s.n,s.k,'N','T');
    run_bench(h,s.m,s.n,s.k,'T','N');
  }
  cublasDestroy(h);
#else
  printf("CUTLASS_ARCH_MMA_SM90_SUPPORTED required\n");
#endif
  return 0;
}
