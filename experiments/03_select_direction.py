# Step 3: compute candidate refusal directions and select the best one on the val set.
# Run cell by cell (VS Code "Run Cell" or copy into Jupyter).

# %%
# 3a: difference-in-means. One candidate direction per (layer, post-instruction position).
from pathlib import Path
import torch

RESULTS = Path(__file__).resolve().parent.parent / "results"
acts = torch.load(RESULTS / "train_acts.pt")                       # each [n_prompts, n_layers, n_pos, d_model]
mean_harmful, mean_harmless = acts["harmful"].mean(0), acts["harmless"].mean(0)  # [28, 5, 1536]
candidates = mean_harmful - mean_harmless                          # r[l, i] = mu_harmful - mu_harmless
print("candidates:", candidates.shape)

# How big is each direction, absolutely and relative to a typical activation at that spot?
abs_norm = candidates.norm(dim=-1)                                 # [28, 5]
rel_norm = abs_norm / mean_harmless.norm(dim=-1)                   # divide out the growth of the residual stream
POS = ["<|im_end|>", "\\n", "<|im_start|>", "assistant", "\\n"]
print(f"{'layer':>5} | " + " | ".join(f"{p:>12}" for p in POS) + "   (abs norm / relative norm)")
for l in range(candidates.shape[0]):
    print(f"{l:>5} | " + " | ".join(f"{a:6.1f} /{r:4.2f}" for a, r in zip(abs_norm[l], rel_norm[l])))

torch.save(candidates, RESULTS / "candidate_dirs.pt")             # 28*5 directions, ~860 KB

# %%
# 3b: tools to TEST a candidate by intervening on the model, scored on the val prompts.
import json
from functools import partial
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device="cpu", dtype=torch.float32)
DATA = RESULTS.parent / "data"
harmful_val = json.loads((DATA / "harmful_val_filtered.json").read_text())    # 26 prompts
harmless_val = json.loads((DATA / "harmless_val_filtered.json").read_text())  # 30 prompts
REFUSAL_TOKS = [40, 2121]  # 'I', 'As' (Step 2b)

def format_chat(instruction: str) -> str:
    return model.tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def last_logits(instructions: list[str], hooks=()) -> torch.Tensor:
    """One batched forward pass. Left padding keeps every prompt's real last token at position -1."""
    prompts = [format_chat(x) for x in instructions]
    return model.run_with_hooks(prompts, fwd_hooks=list(hooks), prepend_bos=False, padding_side="left")[:, -1]

def refusal_scores(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = logits.softmax(-1)[:, REFUSAL_TOKS].sum(-1)            # same log-odds as Step 2b, one per prompt
    return torch.log(p + eps) - torch.log(1 - p + eps)

def kl(base: torch.Tensor, new: torch.Tensor) -> float:
    """KL(base || new) of the next-token distribution, averaged over prompts. 0 = nothing changed."""
    lp, lq = base.log_softmax(-1), new.log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1).mean().item()

# The two interventions, written as TransformerLens hooks (functions that edit an activation mid-forward-pass).
def ablate(act, hook, r_hat):                                  # x - (x . r_hat) r_hat: delete the part of x along r_hat
    return act - (act @ r_hat)[..., None] * r_hat

def add(act, hook, r):                                         # x + r: push every position along r
    return act + r

def ablation_hooks(r: torch.Tensor):
    """Ablate at EVERY layer: the residual stream entering each block and everything attention/MLP write into it."""
    r_hat = r / r.norm()
    names = [f"blocks.{l}.{h}" for l in range(model.cfg.n_layers)
             for h in ("hook_resid_pre", "hook_attn_out", "hook_mlp_out")]
    return [(name, partial(ablate, r_hat=r_hat)) for name in names]

def addition_hooks(r: torch.Tensor, layer: int):
    """Add the raw (un-normalized) direction at ONE layer only: the layer it was extracted from."""
    return [(f"blocks.{layer}.hook_resid_pre", partial(add, r=r))]

# %%
# Baseline (no intervention), then try two hand-picked candidates to see the tools work.
base_harmful, base_harmless = last_logits(harmful_val), last_logits(harmless_val)
print(f"baseline refusal score: harmful {refusal_scores(base_harmful).mean():+.2f}   "
      f"harmless {refusal_scores(base_harmless).mean():+.2f}")

def evaluate(layer: int, pos: int) -> tuple[float, float, float]:
    r = candidates[layer, pos]
    bypass = refusal_scores(last_logits(harmful_val, ablation_hooks(r))).mean().item()    # want LOW
    induce = refusal_scores(last_logits(harmless_val, addition_hooks(r, layer))).mean().item()  # want > 0
    kl_harmless = kl(base_harmless, last_logits(harmless_val, ablation_hooks(r)))         # want < 0.1
    return bypass, induce, kl_harmless

for layer, pos in [(3, -1), (16, -1)]:
    b, i, k = evaluate(layer, pos)
    print(f"layer {layer:2d} pos {pos}:  bypass {b:+.2f}   induce {i:+.2f}   KL {k:.3f}")

# %%
# 3c: run all three tests on every candidate (~40 min on CPU), save the scores.
import time

N_LAYERS = model.cfg.n_layers
SCORES = RESULTS / "direction_scores.json"
scores = json.loads(SCORES.read_text()) if SCORES.exists() else []   # resume if a run was interrupted
done = {s["layer"] for s in scores}
t0 = time.time()
for layer in range(1, N_LAYERS):                     # layer 0 candidates are all zeros (3a): nothing to test
    if layer in done:
        continue
    for p in range(len(POS)):
        b, i, k = evaluate(layer, p)
        scores.append({"layer": layer, "pos": p - len(POS), "token": POS[p], "bypass": b, "induce": i, "kl": k})
    SCORES.write_text(json.dumps(scores, indent=1))  # save after every layer
    print(f"layer {layer:2d} done ({time.time() - t0:.0f}s)", flush=True)

# %%
# Select with the paper's rules: lowest bypass score among candidates that
#   (1) come from layer < 0.8 * n_layers (directions too close to the output tend to just edit the output tokens),
#   (2) actually induce refusal when added (induce > 0),
#   (3) barely disturb harmless prompts when ablated (KL < 0.1).
MAX_LAYER = 0.8 * N_LAYERS
rules = {"layer < 0.8*n_layers": lambda s: s["layer"] < MAX_LAYER,
         "induce > 0":           lambda s: s["induce"] > 0,
         "KL < 0.1":             lambda s: s["kl"] < 0.1}
for name, ok in rules.items():
    print(f"{name:22s} passed by {sum(ok(s) for s in scores):3d}/{len(scores)}")
valid = [s for s in scores if all(ok(s) for ok in rules.values())]
print(f"{'all three':22s} passed by {len(valid):3d}/{len(scores)}\n")

print("bypass score per candidate (* = passes all three rules)")
print(f"{'layer':>5} | " + " | ".join(f"{p:>12}" for p in POS))
for layer in range(1, N_LAYERS):
    row = [s for s in scores if s["layer"] == layer]
    print(f"{layer:>5} | " + " | ".join(f"{s['bypass']:+11.2f}{'*' if s in valid else ' '}" for s in row))

print("\nTop 5 valid candidates:")
for s in sorted(valid, key=lambda s: s["bypass"])[:5]:
    print(f"  layer {s['layer']:2d} pos {s['pos']} {s['token']:>12}: bypass {s['bypass']:+.2f}  induce {s['induce']:+.2f}  KL {s['kl']:.3f}")

best = min(valid, key=lambda s: s["bypass"])
torch.save({**best, "direction": candidates[best["layer"], best["pos"]]}, RESULTS / "refusal_direction.pt")
(RESULTS / "selected_direction.json").write_text(json.dumps(best, indent=1))
print("\nSELECTED:", best)
