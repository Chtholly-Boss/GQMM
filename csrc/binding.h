#pragma once

namespace gqmm::cxx {
void dequant_mxfp4(void *blocks, void *scales, void *out, int num_blocks);
}
