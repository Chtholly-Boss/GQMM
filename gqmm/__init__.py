from . import _C, testing
from torch import Tensor
import torch
import math


def dequant_mxfp4(blocks: Tensor, scales: Tensor) -> Tensor:
    assert blocks.dtype == torch.uint8, "mxfp4 use BYTE storage"
    assert blocks.shape[-1] == 16, (
        "each block should contain exactly 32 mxfp4, i.e. 16 bytes"
    )
    assert scales.dtype == torch.uint8, "scales dtype: UE8"

    *prefix_shape, B = blocks.shape
    num_blocks = math.prod(prefix_shape)

    blocks = blocks.reshape(num_blocks, B)
    scales = scales.reshape(num_blocks, 1)

    out = torch.empty(num_blocks, B * 2, dtype=torch.uint16, device=blocks.device)

    _C.dequant_mxfp4(blocks, scales, out)
    return out.view(torch.bfloat16)


__submodules__ = {"_C", "testing"}

__all__ = list(
    __submodules__
    | {
        "dequant_mxfp4",
    }
)
