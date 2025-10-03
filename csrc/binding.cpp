#include "binding.h"
#include "nanobind/ndarray.h"
#include <cstdint>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void nb_dequant_mxfp4(nb::ndarray<uint8_t> blocks, nb::ndarray<uint8_t> scales,
                      nb::ndarray<uint16_t> out) {
  auto num_blocks = blocks.nbytes() * 2 / 32;
  gqmm::cxx::dequant_mxfp4(blocks.data(), scales.data(), out.data(),
                           num_blocks);
}

NB_MODULE(_C, m) { m.def("dequant_mxfp4", &nb_dequant_mxfp4); }
