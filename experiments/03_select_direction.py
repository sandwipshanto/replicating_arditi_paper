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
