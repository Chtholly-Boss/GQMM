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

    out = torch.empty(num_blocks, B * 2, dtype=torch.bfloat16, device=blocks.device)

    _C.dequant_mxfp4(blocks, scales, out.view(torch.uint16), num_blocks)
    return out


def gemm_bf16_mxfp4(A: Tensor, W: Tensor, scales: Tensor) -> Tensor:
    assert A.is_cuda and W.is_cuda and scales.is_cuda
    assert A.dtype == torch.bfloat16
    assert W.dtype == torch.uint8, "W dtype: UE8 (2 mxfp4 per byte)"
    assert scales.dtype == torch.uint8
    m, k = A.shape
    n, _k = W.shape
    assert k % 32 == 0
    assert _k == k // 2, "W shape must be (n, k // 2)"
    out = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    _C.gemm_bf16_mxfp4(
        A.view(torch.uint16),
        W,
        scales,
        out.view(torch.uint16),
        m,
        n,
        k,
    )
    return out


__submodules__ = {"_C", "testing"}

__all__ = list(
    __submodules__
    | {
        "dequant_mxfp4",
        "gemm_bf16_mxfp4",
    }
)
