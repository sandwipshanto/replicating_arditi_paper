# Research log

## Step 0: Environment check (2026-09-29)

**What we did:** Checked hardware and software before choosing a model.

**Found:**
- Apple M5, 16 GB unified memory, macOS 26.6.2. No NVIDIA GPU; Apple's `mps` backend is the accelerator.
- System Python 3.9.6 only; no torch or TransformerLens installed. Using a fresh Python 3.11 venv.

**Decision:** Qwen2.5-1.5B-Instruct, run locally. About 3 GB in bf16 and about 6 GB in fp32, which fits in 16 GB. If later steps (hundreds of generations) get too slow, move to Colab.

**Learned:**

## Step 1: Load the model, compare a harmful vs. harmless prompt, inspect the residual stream

**Predictions (written before running):**
1. Refuses the bomb prompt? Yes. First words: don't know.
2. Shape of `cache["resid_pre", 0]`: don't know.
3. Last 5 tokens of the formatted prompt: don't know.
4. Residual norm at the last token across layers: grows.

**What happened:**

**Learned:**
