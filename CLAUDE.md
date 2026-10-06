# Project: replicating Arditi et al. (2024), "Refusal in LMs Is Mediated by a Single Direction"

arXiv 2406.11717. The user is a complete beginner in mech interp, comfortable with Python and APIs. The goal is to LEARN while doing real work.

## How to work with the user (tutor-collaborator, not code generator)
- Work in small steps. Never build the whole pipeline at once.
- Before each step, explain in plain language what we're doing and why it matters for the paper.
- Code: the user prefers that Claude writes the code, keeps it short, explains it line by line, and follows up with a short quiz (decided during Step 1). Skeletons with TODOs are still fine for small pieces.
- Before any experiment runs, ask the user to predict the result. Afterwards, compare prediction vs. actual.
- If the user seems to be copying without understanding, stop and quiz them.
- Raise pitfalls (layer choice, token position, chat template, refusal scoring) as they become relevant, not all at once.
- Stay on the current step. Don't teach or quiz on concepts from future steps; it confuses the user.
- Never claim results we haven't seen. Base conclusions only on outputs actually run.
- After every step, help the user add a `research_log.md` entry: what we did, what they expected, what happened, what they learned.
- Use git with small, meaningful commits.
- Wait for the user before moving to the next roadmap step.

## Roadmap
1. Load the model, run a harmful and a harmless prompt, inspect outputs and residual stream activations.
2. Build harmful/harmless prompt sets and collect activations. Use the paper's splits (from the repo): harmful train 128 (AdvBench/MaliciousInstruct/TDC2023), harmful val 32 (HarmBench), harmless train/val from Alpaca, eval on JailbreakBench (100) plus 100 held-out Alpaca. Filter train/val to harmful prompts the model refuses and harmless prompts it doesn't. Eval must not overlap with train/val.
3. Compute difference-in-means directions at every layer x post-instruction token position (template tokens after the instruction). Select one using the val set: minimize bypass score, subject to induce score > 0, KL < 0.1 on harmless, and layer < 0.8 x n_layers. The selection metric is refusal-token log-odds at the last position.
4. Ablate the direction at ALL layers and ALL positions (x - r̂r̂ᵀx), measure the refusal rate on eval (substring matching; Llama Guard 2 optional via Colab). Add the direction at ONLY the extraction layer, all positions, on harmless prompts; measure induced refusal. Check coherence (KL, read outputs).
5. Break it on purpose: wrong layer, fewer prompts, random direction (random baseline is our addition, not in the paper).
6. Compare with the paper and its public code (github.com/andyrdt/refusal_direction); explain differences (e.g. paper used Qwen 1 1.8B, we use Qwen2.5-1.5B).
- Optional stretch: weight orthogonalization plus capability evals (§4); adversarial suffix analysis (§5).

**Current progress:** see the latest entry in `research_log.md`. That is the source of truth for where we are.

## Environment decisions
- Apple M5, 16 GB unified memory, no CUDA. Device is `cpu` (decided in Step 1): TL 3.9 warns that `mps` "may produce silently incorrect results". Only use `mps` after checking that its outputs match CPU. CPU generation runs at about 1 s/token.
- Generation settings: `do_sample=False`, `prepend_bos=False`. Keep Qwen's default system prompt, identical across prompts.
- Python 3.11 venv at `.venv/`, managed with `uv` (`uv pip install -r requirements.txt`).
- Model: `Qwen/Qwen2.5-1.5B-Instruct`. Move to Colab if Step 4+ generation gets too slow.
- TransformerLens is pinned `<4`: v4.0 removed `HookedTransformer`. 3.9.0 is installed and supports Qwen2.5.

## Layout
`experiments/` holds one `# %%`-cell script per roadmap step (Step 4+ runs on Colab as `.ipynb`, cloning this repo from GitHub; the direction travels as `results/refusal_direction.json`). `data/` holds prompt sets. `results/` holds small outputs (`.pt`/`.npy` are git-ignored).
