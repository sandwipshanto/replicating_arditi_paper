# Step 2: build harmful/harmless prompt sets, filter them, collect activations.
# Run cell by cell (VS Code "Run Cell" or copy into Jupyter).

# %%
# 2a: download the paper's prompt splits and sample them exactly like the paper's code.
import json
import random
import urllib.request
from pathlib import Path

REPO = "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset"
DATA = Path(__file__).resolve().parent.parent / "data"

def fetch(path: str) -> list[str]:
    """Download one JSON file from the paper's repo and keep only the instruction text."""
    with urllib.request.urlopen(f"{REPO}/{path}", timeout=30) as r:
        return [d["instruction"] for d in json.load(r)]

# Same seed and same sampling order as the paper's load_and_sample_datasets()
random.seed(42)
harmful_train  = random.sample(fetch("splits/harmful_train.json"), 128)
harmless_train = random.sample(fetch("splits/harmless_train.json"), 128)
harmful_val    = random.sample(fetch("splits/harmful_val.json"), 32)
harmless_val   = random.sample(fetch("splits/harmless_val.json"), 32)
harmless_eval  = random.sample(fetch("splits/harmless_test.json"), 100)
harmful_eval   = fetch("processed/jailbreakbench.json")  # all 100, no sampling

for name, xs in [("harmful_train", harmful_train), ("harmless_train", harmless_train),
                 ("harmful_val", harmful_val), ("harmless_val", harmless_val),
                 ("harmful_eval", harmful_eval), ("harmless_eval", harmless_eval)]:
    print(f"{name:15s} {len(xs):4d}   e.g. {xs[0][:70]!r}")

# %%
# Leakage check: no eval prompt may also appear in train or val.
def norm(s: str) -> str:
    return s.strip().lower()

eval_set = {norm(x) for x in harmful_eval + harmless_eval}
for name, xs in [("harmful_train", harmful_train), ("harmless_train", harmless_train),
                 ("harmful_val", harmful_val), ("harmless_val", harmless_val)]:
    leaks = [x for x in xs if norm(x) in eval_set]
    print(f"{name:15s} overlaps with eval: {len(leaks)}  {leaks}")

# %%
# Drop any leaked prompts from train/val, then save everything to data/.
harmful_train = [x for x in harmful_train if norm(x) not in eval_set]
harmful_val   = [x for x in harmful_val   if norm(x) not in eval_set]

DATA.mkdir(exist_ok=True)
for name, xs in [("harmful_train", harmful_train), ("harmless_train", harmless_train),
                 ("harmful_val", harmful_val), ("harmless_val", harmless_val),
                 ("harmful_eval", harmful_eval), ("harmless_eval", harmless_eval)]:
    (DATA / f"{name}.json").write_text(json.dumps(xs, indent=1))
    print(f"saved data/{name}.json ({len(xs)})")

# %%
