import torch
import gqmm
from gqmm.testing import print_hex

UINT8_LOW, UINT8_HIGH = 0, 255

torch.random.manual_seed(1)


def test_gemm(m, n, k):
    assert k % 32 == 0, "k must be a multiple of 32"
    A = torch.randn(m, k, dtype=torch.bfloat16).cuda()

    W = torch.randint(UINT8_LOW, UINT8_HIGH, (n, k // 2), dtype=torch.uint8).cuda()
    S = torch.randint(0, 128, (n, k // 32), dtype=torch.uint8).cuda()
    # S = torch.ones((n, k // 32), dtype=torch.uint8).cuda() * 127

    W_deq = gqmm.dequant_mxfp4(W.view(-1, 16), S.view(-1)).view(n, k)
    ref = A @ W_deq.t()

    out = gqmm.gemm_bf16_mxfp4(A, W, S)
    diff_mask = (ref - out).abs() > 1e-6
    if diff_mask.any():
        idx = torch.nonzero(diff_mask, as_tuple=False)[0]
        print(
            f"First differing position: {tuple(idx.tolist())}, ref={ref[tuple(idx.tolist())]}, out={out[tuple(idx.tolist())]}"
        )
        assert False, f"GEMM {m}x{n}x{k} failed"
    print(f"GEMM {m}x{n}x{k} passed")


def main():
    test_gemm(16, 8, 32)


if __name__ == "__main__":
    main()
