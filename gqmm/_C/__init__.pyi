from torch import Tensor

def dequant_mxfp4(qweight: Tensor, scales: Tensor, out: Tensor) -> None: ...

__all__ = list(
    {
        "dequant_mxfp4",
    }
)
