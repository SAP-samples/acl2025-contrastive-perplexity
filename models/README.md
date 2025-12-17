# Model Modifications

This directory contains modified versions of HuggingFace transformer models required for contrastive perplexity training.

## Modified Models

### `modeling_llama_modified.py`

Modified version of HuggingFace's LLaMA implementation with support for per-token loss computation.

**Key Modification:**
- Added `loss_reduction` parameter to the `forward()` method (line 1000)
- Modified `CrossEntropyLoss` instantiation to use `reduction=loss_reduction` instead of hardcoded `reduction="mean"` (line 1062)

**Original Source:** `transformers.models.llama.modeling_llama` (HuggingFace Transformers ~4.35.0)

**Why This Modification is Needed:**

The contrastive perplexity objective requires computing perplexity for individual examples, which needs per-token losses. Standard HuggingFace models only return mean-reduced losses.

**Usage:**

Instead of using this file directly, use the `patch_utils.py` utility which provides a cleaner monkey-patch approach:

```python
from transformers import LlamaForCausalLM
from models.patch_utils import patch_causal_lm_for_loss_reduction

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = patch_causal_lm_for_loss_reduction(model)

# Now you can use loss_reduction parameter
output = model(..., loss_reduction="none")
```

## Patch Utility

### `patch_utils.py`

Runtime monkey-patch utility that adds `loss_reduction` parameter support to any HuggingFace CausalLM model.

**Advantages over modified model files:**
- Works with any CausalLM model (LLaMA, Mistral, GPT, etc.)
- No need to maintain full model file copies
- Compatible across different `transformers` versions
- Transparent and easy to understand

See the file for detailed documentation and usage examples.
