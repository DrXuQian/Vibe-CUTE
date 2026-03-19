CUTLASS_DIR := ../3rdparty/cutlass
NVCC        := nvcc
INCLUDES    := -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include
NVCCFLAGS   := -std=c++17 -O3 \
               --generate-code arch=compute_90a,code=[sm_90a] \
               -DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1 \
               --expt-relaxed-constexpr
LDFLAGS     := -lcublas

all: warp_specialized_wgmma_gemm pingpong

warp_specialized_wgmma_gemm: warp_specialized_wgmma_gemm.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

pingpong: warp_specialized_pingpong_gemm.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o warp_specialized_pingpong_gemm $< $(LDFLAGS)

clean:
	rm -f warp_specialized_wgmma_gemm warp_specialized_pingpong_gemm

.PHONY: all clean pingpong
