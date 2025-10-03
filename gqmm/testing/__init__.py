import torch
import numpy
import hexdump
from torch import Tensor


def print_hex(x: Tensor, separate: bool = True):
    hexdump.hexdump(x.cpu().view(torch.int8).numpy().tobytes())
    if separate:
        print(f"{'-' * 80}")


__all__ = list(
    {
        "print_hex",
    }
)
