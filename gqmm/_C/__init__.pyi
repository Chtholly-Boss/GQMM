from torch import Tensor

def dequant_mxfp4(
    qweight: Tensor, scales: Tensor, out: Tensor, num_blocks: int
) -> None: ...
def gemm_bf16_mxfp4(
    A: Tensor, W: Tensor, scales: Tensor, out: Tensor, m: int, n: int, k: int
) -> None: ...

__all__ = list(
    {
        "dequant_mxfp4",
        "gemm_bf16_mxfp4",
    }
)
