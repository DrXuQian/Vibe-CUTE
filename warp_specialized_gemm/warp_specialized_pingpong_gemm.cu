/***************************************************************************************************
 * Pingpong Warp-Specialized WGMMA GEMM — Persistent + Cluster Multicast
 *
 * Single pipeline, interleaved k-block loading, persistent tile-pair scheduling.
 * Producer flows continuously across tile pairs without draining.
 * Cluster (2,1,1) with TMA multicast for B tiles.
 *
 * 384 threads: WG0=Producer, WG1=Consumer0(even stages), WG2=Consumer1(odd stages)
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

template <class EA, class EB, class SLA, class SLB, int S>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<EA, cosize_v<SLA>> smem_A;
  alignas(128) cute::ArrayEngine<EB, cosize_v<SLB>> smem_B;
  typename cutlass::PipelineTmaAsync<S>::SharedStorage pipeline;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta, int Stages, int Log2Swizzle,
          class ClusterShape>
__global__ static __launch_bounds__(384, 1)
void gemm_device_pingpong(ProblemShape shape_MNK, CtaTiler cta_tiler,
                          TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                          TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                          TC* C, CStride dC, TiledMma mma,
                          Alpha alpha, Beta beta, int total_tiles)
{
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
  constexpr int ClusterM = size<0>(ClusterShape{});

  auto tile_to_coord = [&](int idx) -> cute::tuple<int, int, bool> {
    int gnt = SwizzleSize * n_blocks;
    int gidx = idx / gnt, lidx = idx % gnt;
    int mb = gidx * SwizzleSize + lidx % SwizzleSize;
    int nb = lidx / SwizzleSize;
    bool v = (mb < m_blocks && nb < n_blocks);
    return {min(mb, m_blocks - 1), min(nb, n_blocks - 1), v};
  };

  int thread_idx = int(threadIdx.x);
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  int warp_idx_in_wg = warp_idx % cutlass::NumWarpsPerWarpGroup;
  int wg_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
  int lane_pred = cute::elect_one_sync();
  bool is_producer = (warp_group_idx == 0);
  int consumer_idx = warp_group_idx - 1;

  // Cluster info
  uint32_t cluster_local_m = cute::block_rank_in_cluster() % ClusterM;
  uint32_t cluster_local_n = cute::block_rank_in_cluster() / ClusterM;

  if ((warp_idx == 0) && lane_pred) {
    cute::prefetch_tma_descriptor(tma_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(tma_b.get_tma_descriptor());
  }

  // Pipeline: num_consumers=128 (each consumer WG releases its own stages)
  using Pipeline = cutlass::PipelineTmaAsync<Stages>;
  using PipeState = cutlass::PipelineState<Stages>;

  constexpr int tma_bytes = sizeof(TA) * size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{})
                          + sizeof(TB) * size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{});

  typename Pipeline::Params pp;
  pp.transaction_bytes = tma_bytes;
  pp.num_consumers = cutlass::NumThreadsPerWarpGroup;
  pp.is_leader = wg_thread_idx == 0;
  if (is_producer && warp_idx_in_wg == 0)
    pp.role = Pipeline::ThreadCategory::Producer;
  else if (!is_producer)
    pp.role = Pipeline::ThreadCategory::Consumer;
  else
    pp.role = Pipeline::ThreadCategory::NonParticipant;

  Pipeline pipeline(smem.pipeline, pp, ClusterShape{});

  if constexpr (size(ClusterShape{}) > 1) { cute::cluster_sync(); }
  else { __syncthreads(); }

  // B multicast mask
  uint16_t mcast_mask_b = 0;
  {
    auto cl = Layout<ClusterShape>{};
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < ClusterM; ++m)
      mcast_mask_b |= (uint16_t(1) << cl(m, cluster_local_n, Int<0>{}));
  }

  // Non-persistent: 1 tile pair per CTA
  int pair_idx = int(blockIdx.x);
  int t0 = pair_idx * 2, t1 = pair_idx * 2 + 1;
  bool has_t1 = (t1 < total_tiles);

  if (is_producer) {
    cutlass::arch::warpgroup_reg_dealloc<40>();

    if (lane_pred && warp_idx_in_wg == 0) {
      PipeState pw = cutlass::make_producer_start_state<Pipeline>();

      auto [mb0,nb0,v0] = tile_to_coord(t0);
      Tensor gA0 = local_tile(mA, cta_tiler, make_coord(mb0,nb0,_), Step<_1,X,_1>{});
      Tensor gB0 = local_tile(mB, cta_tiler, make_coord(mb0,nb0,_), Step<X,_1,_1>{});
      auto [tAgA0,tAsA0] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                         group_modes<0,2>(sA), group_modes<0,2>(gA0));
      auto [tBgB0,tBsB0] = tma_partition(tma_b, cluster_local_m, make_layout(Int<ClusterM>{}),
                                         group_modes<0,2>(sB), group_modes<0,2>(gB0));

      auto [mb1,nb1,v1] = has_t1 ? tile_to_coord(t1) : tile_to_coord(t0);
      Tensor gA1 = local_tile(mA, cta_tiler, make_coord(mb1,nb1,_), Step<_1,X,_1>{});
      Tensor gB1 = local_tile(mB, cta_tiler, make_coord(mb1,nb1,_), Step<X,_1,_1>{});
      auto [tAgA1,tAsA1] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                         group_modes<0,2>(sA), group_modes<0,2>(gA1));
      auto [tBgB1,tBsB1] = tma_partition(tma_b, cluster_local_m, make_layout(Int<ClusterM>{}),
                                         group_modes<0,2>(sB), group_modes<0,2>(gB1));

      for (int k = 0; k < k_tile_count; ++k) {
        pipeline.producer_acquire(pw);
        auto* bar0 = pipeline.producer_get_barrier(pw);
        copy(tma_a.with(*bar0), tAgA0(_, k), tAsA0(_, pw.index()));
        copy(tma_b.with(*bar0, mcast_mask_b), tBgB0(_, k), tBsB0(_, pw.index()));
        ++pw;
        if (has_t1) {
          pipeline.producer_acquire(pw);
          auto* bar1 = pipeline.producer_get_barrier(pw);
          copy(tma_a.with(*bar1), tAgA1(_, k), tAsA1(_, pw.index()));
          copy(tma_b.with(*bar1, mcast_mask_b), tBgB1(_, k), tBsB1(_, pw.index()));
        }
        ++pw;
      }

      pipeline.producer_tail(pw);
    }

  } else {
    // =================================================================
    // CONSUMER: persistent pingpong with stride-2 stage access
    // =================================================================
    cutlass::arch::warpgroup_reg_alloc<232>();

    ThrMMA thr_mma = mma.get_thread_slice(wg_thread_idx);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    constexpr int K_PIPE_MMAS = 1;

    PipeState rd;
    if (consumer_idx == 1) { ++rd; }
    PipeState rl = rd;

    int my_tile = t0 + consumer_idx;  // pair_idx * 2 + consumer_idx
    bool should_run = (consumer_idx == 0) || has_t1;

    if (should_run) {
      auto [mb, nb, valid] = tile_to_coord(my_tile);
      Tensor gC_t = local_tile(mC, cta_tiler, make_coord(mb, nb, _), Step<_1,_1,X>{});
      Tensor tCgC = thr_mma.partition_C(gC_t);
      Tensor tCrC = thr_mma.make_fragment_C(tCgC);

      // Prologue
      {
        auto bt = pipeline.consumer_try_wait(rd);
        pipeline.consumer_wait(rd, bt);
        int rs = rd.index();
        mma.accumulate_ = GMMA::ScaleOut::Zero;
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        CUTLASS_PRAGMA_UNROLL
        for (int kb = 0; kb < size<2>(tCrA); ++kb) {
          cute::gemm(mma, tCrA(_,_,kb,rs), tCrB(_,_,kb,rs), tCrC);
          mma.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        ++rd; ++rd;
      }
      warpgroup_fence_operand(tCrC);

      // Main loop
      CUTE_NO_UNROLL
      for (int k = 1; k < k_tile_count; ++k) {
        auto bt = pipeline.consumer_try_wait(rd);
        pipeline.consumer_wait(rd, bt);
        int rs = rd.index();
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        cute::gemm(mma, tCrA(_,_,_,rs), tCrB(_,_,_,rs), tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<K_PIPE_MMAS>();
        warpgroup_fence_operand(tCrC);
        pipeline.consumer_release(rl);
        ++rd; ++rd;
        ++rl; ++rl;
      }

      warpgroup_wait<0>();
      for (int i = 0; i < K_PIPE_MMAS; ++i) {
        pipeline.consumer_release(rl);
        ++rl; ++rl;
      }

      if (valid) { axpby(alpha, tCrC, beta, tCgC); }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt(int m, int n, int k,
             Alpha alpha, TA const* A, int ldA, TB const* B, int ldB,
             Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  auto M=int(m); auto N=int(n); auto K=int(k);
  auto prob_shape=make_shape(M,N,K);
  auto dA=make_stride(Int<1>{},ldA); auto dB=make_stride(Int<1>{},ldB); auto dC=make_stride(Int<1>{},ldC);
  auto bM=Int<128>{}; auto bN=Int<128>{}; auto bK=Int<64>{};
  auto cta_tiler=make_shape(bM,bN,bK);
  constexpr int Stages=7, Log2Swizzle=3, SwizzleSize=(1<<Log2Swizzle);
  constexpr int ClusterM=2, ClusterN=1;
  using ClusterShape = Shape<Int<ClusterM>, Int<ClusterN>, _1>;

  auto sA=tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{},make_shape(bM,bK,Int<Stages>{}));
  auto sB=tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{},make_shape(bN,bK,Int<Stages>{}));
  TiledMMA mma=make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});

  Tensor mA_h=make_tensor(A,make_shape(M,K),dA); Tensor mB_h=make_tensor(B,make_shape(N,K),dB);
  Copy_Atom tmaA=make_tma_atom(SM90_TMA_LOAD{},mA_h,sA(_,_,0),make_shape(bM,bK));
  Copy_Atom tmaB=make_tma_atom(SM90_TMA_LOAD_MULTICAST{},mB_h,sB(_,_,0),make_shape(bN,bK),Int<ClusterM>{});

  int smem_size=int(sizeof(SharedStorage<TA,TB,decltype(sA),decltype(sB),Stages>));
  int m_blocks=size(ceil_div(m,bM)); int n_blocks=size(ceil_div(n,bN));
  int total_tiles=((m_blocks+SwizzleSize-1)/SwizzleSize)*SwizzleSize*n_blocks;
  total_tiles=((total_tiles+ClusterM-1)/ClusterM)*ClusterM;
  int total_pairs=(total_tiles+1)/2;

  // Non-persistent: 1 CTA per tile pair, round to cluster
  int grid_blocks = ((total_pairs + ClusterM - 1) / ClusterM) * ClusterM;

  dim3 dimBlock(384); dim3 dimCluster(ClusterM,ClusterN,1); dim3 dimGrid(grid_blocks,1,1);
  auto* kp=&gemm_device_pingpong<decltype(prob_shape),decltype(cta_tiler),
      TA,decltype(sA),decltype(tmaA),TB,decltype(sB),decltype(tmaB),
      TC,decltype(dC),decltype(mma),decltype(alpha),decltype(beta),Stages,Log2Swizzle,ClusterShape>;
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kp,cudaFuncAttributeMaxDynamicSharedMemorySize,smem_size));
  cutlass::ClusterLaunchParams params={dimGrid,dimBlock,dimCluster,smem_size};
  cutlass::launch_kernel_on_cluster(params,(void const*)kp,
      prob_shape,cta_tiler,A,tmaA,B,tmaB,C,dC,mma,alpha,beta,total_tiles);
  CUTE_CHECK_LAST();
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn(int m, int n, int k,
             Alpha alpha, TA const* A, int ldA, TB const* B, int ldB,
             Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
  auto M=int(m); auto N=int(n); auto K=int(k);
  auto prob_shape=make_shape(M,N,K);
  auto dA=make_stride(ldA,Int<1>{}); auto dB=make_stride(ldB,Int<1>{}); auto dC=make_stride(Int<1>{},ldC);
  auto bM=Int<128>{}; auto bN=Int<128>{}; auto bK=Int<64>{};
  auto cta_tiler=make_shape(bM,bN,bK);
  constexpr int Stages=7, Log2Swizzle=3, SwizzleSize=(1<<Log2Swizzle);
  constexpr int ClusterM=2, ClusterN=1;
  using ClusterShape = Shape<Int<ClusterM>, Int<ClusterN>, _1>;

  auto sA=tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{},make_shape(bM,bK,Int<Stages>{}));
  auto sB=tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{},make_shape(bN,bK,Int<Stages>{}));
  TiledMMA mma=make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});

  Tensor mA_h=make_tensor(A,make_shape(M,K),dA); Tensor mB_h=make_tensor(B,make_shape(N,K),dB);
  Copy_Atom tmaA=make_tma_atom(SM90_TMA_LOAD{},mA_h,sA(_,_,0),make_shape(bM,bK));
  Copy_Atom tmaB=make_tma_atom(SM90_TMA_LOAD_MULTICAST{},mB_h,sB(_,_,0),make_shape(bN,bK),Int<ClusterM>{});

  int smem_size=int(sizeof(SharedStorage<TA,TB,decltype(sA),decltype(sB),Stages>));
  int m_blocks=size(ceil_div(m,bM)); int n_blocks=size(ceil_div(n,bN));
  int total_tiles=((m_blocks+SwizzleSize-1)/SwizzleSize)*SwizzleSize*n_blocks;
  total_tiles=((total_tiles+ClusterM-1)/ClusterM)*ClusterM;
  int total_pairs=(total_tiles+1)/2;

  cudaDeviceProp props; cudaGetDeviceProperties(&props,0);
  int persistent = (props.multiProcessorCount / ClusterM) * ClusterM;
  int grid_blocks = min(total_pairs, persistent);
  grid_blocks = ((grid_blocks + ClusterM - 1) / ClusterM) * ClusterM;

  dim3 dimBlock(384); dim3 dimCluster(ClusterM,ClusterN,1); dim3 dimGrid(grid_blocks,1,1);
  auto* kp=&gemm_device_pingpong<decltype(prob_shape),decltype(cta_tiler),
      TA,decltype(sA),decltype(tmaA),TB,decltype(sB),decltype(tmaB),
      TC,decltype(dC),decltype(mma),decltype(alpha),decltype(beta),Stages,Log2Swizzle,ClusterShape>;
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kp,cudaFuncAttributeMaxDynamicSharedMemorySize,smem_size));
  cutlass::ClusterLaunchParams params={dimGrid,dimBlock,dimCluster,smem_size};
  cutlass::launch_kernel_on_cluster(params,(void const*)kp,
      prob_shape,cta_tiler,A,tmaA,B,tmaB,C,dC,mma,alpha,beta,total_tiles);
  CUTE_CHECK_LAST();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(char tA, char tB, int m, int n, int k,
          Alpha a, TA const* A, int ldA, TB const* B, int ldB,
          Beta b, TC* C, int ldC, cudaStream_t s = 0) {
  if (tA=='N'&&tB=='T') return gemm_nt(m,n,k,a,A,ldA,B,ldB,b,C,ldC,s);
  if (tA=='T'&&tB=='N') return gemm_tn(m,n,k,a,A,ldA,B,ldB,b,C,ldC,s);
  assert(false);
}

void cublas_gemm(cublasHandle_t h, int m, int n, int k,
                 cute::half_t const* A, cute::half_t const* B, cute::half_t* C,
                 char tA, char tB) {
  __half a16=__float2half(1.f), b16=__float2half(0.f);
  cublasHgemm(h,(tA=='N')?CUBLAS_OP_N:CUBLAS_OP_T,(tB=='N')?CUBLAS_OP_N:CUBLAS_OP_T,
              m,n,k,&a16,(__half const*)A,(tA=='N')?m:k,
              (__half const*)B,(tB=='N')?k:n,&b16,(__half*)C,m);
}

double max_rel_err(thrust::host_vector<cute::half_t> const& r,
                   thrust::host_vector<cute::half_t> const& ref, int sz) {
  double mx=0; for(int i=0;i<sz;i++){double a=float(r[i]),b=float(ref[i]);
  mx=std::max(mx,std::abs(a-b)/std::max(std::abs(b),1e-5));} return mx;
}

bool run_test(cublasHandle_t h, int m, int n, int k, char tA, char tB) {
  using T=cute::half_t;
  thrust::host_vector<T> hA(m*k),hB(n*k),hC(m*n,T(0));
  srand(42); for(auto&v:hA)v=T(float((rand()%5)-2)); for(auto&v:hB)v=T(float((rand()%5)-2));
  thrust::device_vector<T> dA=hA,dB=hB,dC1=hC,dC2=hC;
  cublas_gemm(h,m,n,k,dA.data().get(),dB.data().get(),dC1.data().get(),tA,tB);
  cudaDeviceSynchronize();
  gemm(tA,tB,m,n,k,T(1.f),dA.data().get(),(tA=='N')?m:k,dB.data().get(),(tB=='N')?k:n,T(0.f),dC2.data().get(),m);
  CUTE_CHECK_LAST(); cudaDeviceSynchronize();
  double e=max_rel_err(thrust::host_vector<T>(dC2),thrust::host_vector<T>(dC1),m*n);
  bool ok=e<1e-2; printf("  [%c%c] M=%4d N=%4d K=%4d  err=%.2e  %s\n",tA,tB,m,n,k,e,ok?"PASS":"FAIL");
  return ok;
}

void run_bench(cublasHandle_t h, int m, int n, int k, char tA, char tB) {
  using T=cute::half_t;
  thrust::host_vector<T> hA(m*k),hB(n*k),hC(m*n,T(0));
  for(auto&v:hA)v=T(float((rand()%5)-2)); for(auto&v:hB)v=T(float((rand()%5)-2));
  thrust::device_vector<T> dA=hA,dB=hB,dC=hC;
  double gf=2.0*m*n*k*1e-9; GPU_Clock timer;
  int la=(tA=='N')?m:k, lb=(tB=='N')?k:n;
  for(int i=0;i<10;i++) gemm(tA,tB,m,n,k,T(1.f),dA.data().get(),la,dB.data().get(),lb,T(0.f),dC.data().get(),m);
  cudaDeviceSynchronize(); timer.start();
  for(int i=0;i<100;i++) gemm(tA,tB,m,n,k,T(1.f),dA.data().get(),la,dB.data().get(),lb,T(0.f),dC.data().get(),m);
  double ct=timer.seconds()/100; CUTE_CHECK_LAST();
  __half a16=__float2half(1.f),b16=__float2half(0.f);
  auto opA=(tA=='N')?CUBLAS_OP_N:CUBLAS_OP_T,opB=(tB=='N')?CUBLAS_OP_N:CUBLAS_OP_T;
  for(int i=0;i<10;i++) cublasHgemm(h,opA,opB,m,n,k,&a16,(__half*)dA.data().get(),(tA=='N')?m:k,
    (__half*)dB.data().get(),(tB=='N')?k:n,&b16,(__half*)dC.data().get(),m);
  cudaDeviceSynchronize(); timer.start();
  for(int i=0;i<100;i++) cublasHgemm(h,opA,opB,m,n,k,&a16,(__half*)dA.data().get(),(tA=='N')?m:k,
    (__half*)dB.data().get(),(tB=='N')?k:n,&b16,(__half*)dC.data().get(),m);
  double bt=timer.seconds()/100;
  printf("  [%c%c] M=%4d N=%4d K=%4d  | PP: %7.1f GF/s (%.4fms) | cuBLAS: %7.1f GF/s (%.4fms) | %.1f%%\n",
         tA,tB,m,n,k,gf/ct,ct*1e3,gf/bt,bt*1e3,(gf/ct)/(gf/bt)*100);
}

int main(int argc, char** argv) {
  cudaDeviceProp props; int dev; cudaGetDevice(&dev); cudaGetDeviceProperties(&props,dev);
  if(props.major!=9){printf("Need SM90\n");return 0;}
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  printf("=== Pingpong GEMM (Persistent + Cluster Multicast) ===\n");
  printf("GPU: %s\n\n",props.name);
  cublasHandle_t h; cublasCreate(&h);
  printf("--- Correctness ---\n"); bool ok=true;
  struct S{int m,n,k;};
  for(auto s:std::vector<S>{{128,128,64},{256,256,256},{512,512,512},{1024,2048,512},{2048,2048,2048},{4096,4096,4096}}){
    ok&=run_test(h,s.m,s.n,s.k,'N','T'); ok&=run_test(h,s.m,s.n,s.k,'T','N');
  }
  printf("\nOverall: %s\n\n",ok?"ALL PASSED":"SOME FAILED");
  printf("--- Performance (100 iters) ---\n");
  for(auto s:std::vector<S>{{1024,1024,1024},{2048,2048,2048},{4096,4096,4096},{8192,8192,8192}}){
    run_bench(h,s.m,s.n,s.k,'N','T'); run_bench(h,s.m,s.n,s.k,'T','N');
  }
  cublasDestroy(h);
#else
  printf("Need CUTLASS_ARCH_MMA_SM90_SUPPORTED\n");
#endif
  return 0;
}
