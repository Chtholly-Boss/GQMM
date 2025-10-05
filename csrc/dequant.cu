#include "binding.h"
#include "dequant.cuh"
#include <cstdint>
#include <cstdio>
#include "helper.cuh"

static __global__ void k_dequant_mxfp4(uint32_t *blocks, uint8_t *scales,
                                       uint32_t *out, int num_blocks) {
  // each thread process a block
  auto const gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid >= num_blocks)
    return;

  // mxfp4 use fixed group size of 32, 1 ld.dwordx4 = 128 bits = 32 x 4-bit
  uint32_t block[4];
  auto block_ptr = reinterpret_cast<uint4 *>(blocks) + gid;
  v_mov<uint4>(block, block_ptr);

  // 1 scale per 32 elements, 4 scales packed in a uint32_t
  uint32_t scale = scales[gid];
  auto out_ptr = reinterpret_cast<nv_bfloat16 *>(out) + gid * 32;

  nv_bfloat16 v8dq[8];
  for (int i = 0; i < 4; i++) {
    uint32_t v8q = block[i];
    for (int j = 0; j < 8; j++) {
      uint32_t v4q = (v8q >> (j * 4)) & 0xF;
      v8dq[j] = dequant_mxfp4_with_scale(v4q, scale);
    }
    v_mov<uint4>(out_ptr + i * 8, v8dq);
  }
}

namespace gqmm::cxx {
void dequant_mxfp4(void *blocks, void *scales, void *out, int num_blocks) {
  k_dequant_mxfp4<<<ceil_div(num_blocks, 32), 32>>>(
      reinterpret_cast<uint32_t *>(blocks), reinterpret_cast<uint8_t *>(scales),
      reinterpret_cast<uint32_t *>(out), num_blocks);
}
} // namespace gqmm::cxx
