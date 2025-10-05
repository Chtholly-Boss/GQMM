#include "binding.h"
#include "nanobind/ndarray.h"
#include <cstdint>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void nb_dequant_mxfp4(nb::ndarray<uint8_t> blocks, nb::ndarray<uint8_t> scales,
                      nb::ndarray<uint16_t> out, int num_blocks) {
  gqmm::cxx::dequant_mxfp4(blocks.data(), scales.data(), out.data(),
                           num_blocks);
}

void nb_gemm_bf16_mxfp4(nb::ndarray<uint16_t> A, nb::ndarray<uint8_t> W,
                        nb::ndarray<uint8_t> scales, nb::ndarray<uint16_t> out,
                        int m, int n, int k) {
  gqmm::cxx::gemm_bf16_mxfp4(A.data(), W.data(), scales.data(), out.data(), m,
                             n, k);
}

NB_MODULE(_C, m) {
  m.def("dequant_mxfp4", &nb_dequant_mxfp4);
  m.def("gemm_bf16_mxfp4", &nb_gemm_bf16_mxfp4);
}
