# Model parallel inference
# Note: This script is for demonstration purposes only. It is not designed for production use.
#       See gpt_oss.chat for a more complete example with the Harmony parser.
# torchrun --nproc-per-node=4 -m gpt_oss.generate -p "why did the chicken cross the road?" model/

import argparse

from gpt_oss.tokenizer import get_tokenizer


def main(args):
    # use torch as backend
    from gpt_oss.torch.utils import init_distributed
    from gpt_oss.torch.model import TokenGenerator as TorchGenerator
    device = init_distributed()
    generator = TorchGenerator(args.checkpoint, device=device)

    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(args.prompt)
    max_tokens = args.limit
    for token, logprob in generator.generate(tokens, stop_tokens=[tokenizer.eot_token], temperature=args.temperature, max_tokens=max_tokens, return_logprobs=True):
        tokens.append(token)
        token_text = tokenizer.decode([token])
        print(
            f"Generated token: {repr(token_text)}, logprob: {logprob}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text generation example")
    parser.add_argument(
        "checkpoint",
        metavar="FILE",
        type=str,
        help="Path to the SafeTensors checkpoint",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        metavar="PROMPT",
        type=str,
        default="How are you?",
        help="LLM prompt",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        metavar="TEMP",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "-l",
        "--limit",
        metavar="LIMIT",
        type=int,
        default=0,
        help="Limit on the number of tokens (0 to disable)",
    )
    args = parser.parse_args()

    main(args)
