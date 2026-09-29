# Research log

## Step 0: Environment check (2026-09-29)

**What we did:** Checked hardware and software before choosing a model.

**Found:**
- Apple M5, 16 GB unified memory, macOS 26.6.2. No NVIDIA GPU; Apple's `mps` backend is the accelerator.
- System Python 3.9.6 only; no torch or TransformerLens installed. Using a fresh Python 3.11 venv.

**Decision:** Qwen2.5-1.5B-Instruct, run locally. About 3 GB in bf16 and about 6 GB in fp32, which fits in 16 GB. If later steps (hundreds of generations) get too slow, move to Colab.

**Learned:** TransformerLens 4.0 removed `HookedTransformer`, so it is pinned `<4` (3.9.0). TL warns that `mps` "may produce silently incorrect results", so we switched to CPU.

## Step 1: Load the model, compare a harmful vs. harmless prompt, inspect the residual stream

**Predictions (written before running):**
1. Refuses the bomb prompt? Yes. First words: don't know.
2. Shape of `cache["resid_pre", 0]`: don't know.
3. Last 5 tokens of the formatted prompt: don't know.
4. Residual norm at the last token across layers: grows.
5. (Added later) Layer where harmful and harmless last-token norms start to differ: near the end.

**Setup decisions:** CPU, fp32, greedy decoding (`do_sample=False`, since TL defaults to sampling), `prepend_bos=False` (TL's BOS token for Qwen is `<|im_end|>`, so adding it would be unnatural). Qwen's template auto-inserts a system prompt ("You are Qwen, created by Alibaba Cloud...") that we kept for now. It must stay identical across all prompts.

**What happened:**
1. ✅ Refused: "I'm sorry, but I can't assist with that." Stopped by itself after 12 tokens. The paper airplane prompt complied ("Sure, here are the step-by-step instructions...") and ran to the 60-token limit. Generation speed on CPU was about 1 s/token.
2. ✅ Shape `[1, 41, 1536]` = [batch, position, d_model]. Both prompts are 41 tokens and differ only at positions 33–34.
3. Last 5 tokens: `<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`. They are identical for both prompts, yet they lead to opposite behaviour.
4. ✅ Last-token norm grows from 1.21 (layer 0) to ~280 (layer 27). Fast growth in layers 0–5, flat around 34–39 in layers 6–12, steep growth in layers 20–27.
5. ❌ Partly. The *norms* differ noticeably from layer ~14 (the middle), with the largest relative gap at layers 17–20 (up to ~21%), not just at the end.
6. Direction check (`(a-b).norm()` and cosine): the vectors differ from layer 1 (diff 0.38) even when their norms match (layer 5: norms 29.34 vs 29.24, diff 1.56). There's a sharp split at layers 14–17 (cos 0.995 → 0.848, about 6° → 32°), and the minimum is cos 0.757 (~41°) at layer 23. Cosine rises again at layers 26–27 as both vectors grow in a shared direction.
- Layer 0 embeddings are identical for identical tokens, because no mixing has happened yet.
- `generate` returns text with special tokens stripped, so never re-tokenize its output for activation analysis.

**Caveat:** this is one pair of prompts, and they differ in topic (explosives vs. crafts) as well as harmfulness. We can't yet say the mid-layer split is about *harmfulness*. Step 2 uses many prompts to address this.

**Learned (in my words):** With a prompt as input, it gets processed in each layer. The vectors update with every layer. To understand when in the layers the model starts to think of 2 things apart, we can derive their differences and their difference of direction at each layer.
- Norm only measures length. Two vectors with the same length can still point in different directions, so use diff_norm and cosine too. diff_norm = 0 and cosine = 1 both mean "identical", on different scales.
