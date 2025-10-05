#pragma once

namespace gqmm::cxx {
void dequant_mxfp4(void *blocks, void *scales, void *out, int num_blocks);
void gemm_bf16_mxfp4(void *A, void *W, void *scales, void *out, int m, int n,
                     int k);
} // namespace gqmm::cxx
