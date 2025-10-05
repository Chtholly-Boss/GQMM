import torch
from torch import Tensor
import gqmm
import math
from gqmm.testing import print_hex

torch.manual_seed(0)

UINT8_LOW, UINT8_HIGH = 0, 255

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def dequant_mxfp4_torch(blocks: Tensor, scales: Tensor) -> Tensor:
    assert blocks.dtype == torch.uint8 and blocks.shape[-1] == 16
    assert scales.dtype == torch.uint8
    assert blocks.shape[:-1] == scales.shape

    scales = scales.to(torch.int32) - 127
    lut = torch.tensor(FP4_VALUES, dtype=torch.bfloat16, device=blocks.device)

    *prefix_shape, B = blocks.shape
    num_blocks = math.prod(prefix_shape)

    blocks = blocks.reshape(num_blocks, B)
    scales = scales.reshape(num_blocks, 1)

    out = torch.empty(num_blocks, B * 2, dtype=torch.bfloat16, device=blocks.device)

    # nibble indices -> int64
    idx_lo = (blocks & 0x0F).to(torch.long)
    idx_hi = (blocks >> 4).to(torch.long)

    out[:, 0::2] = lut[idx_lo]
    out[:, 1::2] = lut[idx_hi]
    torch.ldexp(out, scales, out=out)
    return out


def test_dequant_mxfp4(num_blocks):
    # random mxfp4 simulation
    blocks = torch.randint(
        UINT8_LOW, UINT8_HIGH, (num_blocks, 16), dtype=torch.uint8, device="cuda"
    )
    scales = torch.randint(
        UINT8_LOW,
        UINT8_HIGH,
        (num_blocks,),
        dtype=torch.uint8,
        device="cuda",
    )
    ref = dequant_mxfp4_torch(blocks, scales)
    out = gqmm.dequant_mxfp4(blocks, scales)
    # print_hex(out)
    # print_hex(ref)
    assert torch.allclose(ref, out), f"mxfp4 dequant failed for {num_blocks} blocks"


def main():
    for num_blocks in [1, 4, 16, 64, 256, 1024, 2048, 4096]:
        test_dequant_mxfp4(num_blocks)
        print(f">> mxfp4 dequant passed for {num_blocks} blocks")


if __name__ == "__main__":
    main()
