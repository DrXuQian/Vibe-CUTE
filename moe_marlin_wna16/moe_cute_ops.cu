/*
 * MOE CuTe Marlin kernel — standalone pybind11 binding.
 * Provides moe_marlin_cute_gemm() that wraps MarlinCuteMoe kernel.
 */

#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "marlin_cute_moe_kernel.cuh"

torch::Tensor moe_marlin_cute_gemm(
    torch::Tensor& a,                    // (M, K) FP16 input
    torch::Tensor& b_q_weight,           // (num_experts, K/16, N*16/8) INT4 packed
    torch::Tensor& b_scales,             // (num_experts, K/groupsize, N) or (num_experts, 1, N)
    torch::Tensor& sorted_token_ids,     // from moe_align_block_size
    torch::Tensor& expert_ids,           // expert id per MOE block
    torch::Tensor& num_tokens_post_padded,
    torch::Tensor& topk_weights,
    torch::Tensor& workspace,
    int64_t top_k,
    bool mul_topk_weights,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    int64_t groupsize
) {
    int64_t thread_k = -1, thread_n = -1;
    int dev = a.get_device();
    int sms = -1;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

    // Allocate output: (size_m * top_k, size_n) — one row per (token, expert) pair
    auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
    torch::Tensor c = torch::zeros({size_m * top_k, size_n}, options);

    int err = marlin_cuda_moe_cute(
        a.data_ptr(),
        b_q_weight.data_ptr(),
        c.data_ptr(),
        b_scales.data_ptr(),
        sorted_token_ids.data_ptr<int32_t>(),
        expert_ids.data_ptr<int32_t>(),
        num_tokens_post_padded.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        top_k,
        mul_topk_weights,
        size_m,
        size_n,
        size_k,
        c.data_ptr(),  // workspace (reuse)
        groupsize,
        dev,
        at::cuda::getCurrentCUDAStream(dev),
        thread_k,
        thread_n,
        sms
    );

    if (err == ERR_PROB_SHAPE) {
        AT_ERROR("Problem shape not compatible with thread config");
    } else if (err == ERR_KERN_SHAPE) {
        AT_ERROR("No CuTe MOE kernel for this thread config / groupsize");
    }

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_marlin_cute_gemm", &moe_marlin_cute_gemm,
          "MOE Marlin CuTe GEMM");
}
