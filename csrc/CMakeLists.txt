set(CUDA_SRCS 
    dequant.cu
    gemm.cu
)

nanobind_add_module(_C binding.cpp ${CUDA_SRCS}) # add extension module
