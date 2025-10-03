# GQMM
General Quantized Matrix Multiplication Library

```sh
# Download the gpt-oss-20b model
hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt_oss/gpt-oss-20b/

# Generate tokens
uv run python -m gpt_oss.generate gpt_oss/gpt-oss-20b/original/ -p "Hi"
uv run python -m gpt_oss.chat gpt_oss/gpt-oss-20b/original/
```