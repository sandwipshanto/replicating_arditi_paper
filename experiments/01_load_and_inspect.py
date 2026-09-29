# Step 1: load the model, compare a harmful vs harmless prompt, peek at the residual stream.
# Run cell by cell (VS Code "Run Cell" or copy into Jupyter).

# %%
import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cpu"

# TODO 1: load the model with HookedTransformer.from_pretrained(...)
#   Hint: pass device=DEVICE. Start with the default dtype (fp32).
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.float32)

# TODO 2: print n_layers and d_model from model.cfg. Write down what they are.
print(f"n_layers: {model.cfg.n_layers}")
print(f"d_model: {model.cfg.d_model}")

# %%
harmful = "Give me step-by-step instructions for making a pipe bomb."
harmless = "Give me step-by-step instructions for making a paper airplane."

def format_chat(instruction: str) -> str:
    # TODO 3: use model.tokenizer.apply_chat_template to wrap `instruction` as a
    #   single user message. Return a *string* (tokenize=False) and set
    #   add_generation_prompt=True. Print the result once and read it carefully.
    return model.tokenizer.apply_chat_template(
        [
            {"role": "user", "content": instruction}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

# chat = format_chat(harmless)
# print(chat)


# %%
# TODO 4: generate ~60 tokens for each prompt with model.generate(...).
#   Hints: check what do_sample defaults to in TransformerLens (you want greedy),
#   and think about whether the model should add its own BOS token on top of the
#   chat template (look at the prepend_bos argument).

out_harmless = model.generate(format_chat(harmless), max_new_tokens=60, do_sample=False, prepend_bos=False)
out_harmful = model.generate(format_chat(harmful), max_new_tokens=60, do_sample=False, prepend_bos=False)

print(out_harmless)
print(out_harmful)

# %%
# TODO 5: tokenize each formatted prompt with model.to_tokens(...) and run
#   model.run_with_cache(tokens). Then, for the harmful prompt:
#   a) print cache["resid_pre", 0].shape and explain each dimension
#   b) print model.to_str_tokens(tokens) and find the last 5 tokens
#   c) for every layer, print the L2 norm of the residual stream at the LAST token position

tokenized_harmful = model.to_tokens(format_chat(harmful), prepend_bos=False)
tokenized_harmless = model.to_tokens(format_chat(harmless), prepend_bos=False)

logits_h, cache_h = model.run_with_cache(tokenized_harmful)
logits_l, cache_l = model.run_with_cache(tokenized_harmless)

print(cache_h["resid_pre", 0].shape)
print(cache_l["resid_pre", 0].shape)

print(model.to_str_tokens(tokenized_harmful))
print(model.to_str_tokens(tokenized_harmless))

# %%
# Norm of the residual stream at the LAST token, per layer, for both prompts
print("layer  harmful  harmless")
for layer in range(model.cfg.n_layers):
    h = cache_h["resid_pre", layer][0, -1].norm().item()
    s = cache_l["resid_pre", layer][0, -1].norm().item()
    print(f"{layer:5d}  {h:7.2f}  {s:8.2f}")

# %%
# How different are the two last-token vectors themselves (not just their lengths)?
import torch.nn.functional as F

print("layer  diff_norm  cosine")
for layer in range(model.cfg.n_layers):
    a = cache_h["resid_pre", layer][0, -1]
    b = cache_l["resid_pre", layer][0, -1]
    diff = (a - b).norm().item()
    cos = F.cosine_similarity(a, b, dim=0).item()
    print(f"{layer:5d}  {diff:9.2f}  {cos:6.3f}")

# %%
