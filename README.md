# Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2401.08491-29d634.svg)](https://arxiv.org/abs/2401.08491)
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/acl2025-contrastive-perplexity)](https://api.reuse.software/info/github.com/SAP-samples/acl2025-contrastive-perplexity)


#### News
- **12/18/2025:** :confetti_ball: Model provided :tada:
- 12/17/2025: Source code provided 
- 05/30/2025: Initial repo created

This repository contains the source code for the [ACL2 025](https://2025.aclweb.org/) paper: [**Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models**](https://arxiv.org/abs/2401.08491)

### Abstract
The generation of toxic content by large language models (LLMs) remains a critical challenge for the safe deployment of language technology. We propose a novel framework for implicit knowledge editing and controlled text generation by fine-tuning LLMs with a prototype-based contrastive perplexity objective. Central to our method is the construction of hard negatives—toxic outputs that are generated through adversarial paraphrasing to be semantically similar and closely matched in length and model probability to their non-toxic counterparts. By training on these challenging and realistic pairs, our approach ensures robust and stable contrastive optimization. Experimental results in the domain of detoxification demonstrate that our method significantly reduces toxic generation while maintaining strong performance on downstream tasks such as commonsense reasoning and reading comprehension. Our findings highlight the effectiveness of leveraging hard negatives for attribute-aware language model fine-tuning.

#### Authors:
 - [Tassilo Klein](https://tjklein.github.io/)
 - [Moin Nabi](https://moinnabi.github.io/)


## Language Models

Language models trained for which the performance is reported in the paper are available at the [Huggingface Model Repository](https://huggingface.co/models):
 - [https://huggingface.co/TJKlein/ContrastivePerplexity](https://huggingface.co/TJKlein/ContrastivePerplexity)

   
## Description

The generation of toxic content by large language models (LLMs) remains a critical challenge for the safe deployment of language technology. We propose a novel framework for implicit knowledge editing and controlled text generation by fine-tuning LLMs with a **prototype-based contrastive perplexity objective**. 

Central to our method is the construction of **hard negatives**—toxic outputs that are generated through adversarial paraphrasing to be semantically similar and closely matched in length and model probability to their non-toxic counterparts. By training on these challenging and realistic pairs, our approach ensures robust and stable contrastive optimization.

### Key Contributions

- **Contrastive Perplexity Framework**: A novel training objective that leverages perplexity differences between toxic and non-toxic text pairs
- **Hard Negative Mining**: Adversarial paraphrasing technique to generate challenging negative examples
- **Effective Detoxification**: Significantly reduces toxic generation while maintaining strong performance on downstream tasks
- **Minimal Performance Degradation**: Preserves model capabilities on commonsense reasoning and reading comprehension

## Requirements

### Environment

- Python 3.8+
- CUDA-capable GPU (tested on NVIDIA A100)
- 16GB+ GPU memory (for 4-bit quantization) or 40GB+ (for fp16)

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Main dependencies:
- `transformers >= 4.35.0`
- `torch >= 2.0.0`
- `peft >= 0.6.0`
- `datasets >= 2.14.0`
- `bitsandbytes >= 0.41.0`
- `einops >= 0.7.0`
- `wandb` (optional, for experiment tracking)

## Installation

```bash
# Clone the repository
git clone https://github.com/SAP-samples/acl2025-contrastive-perplexity.git
cd acl2025-contrastive-perplexity

# Install dependencies
pip install -r requirements.txt

# Prepare data (downloads ToxiGen dataset)
bash scripts/prepare_data.sh
```

## Critical: Model Patching for Loss Reduction

> **⚠️ IMPORTANT**: The contrastive perplexity method requires **per-token loss computation**, which is not supported in standard HuggingFace transformer implementations. We provide a utility to patch models at runtime.

### Why Patching is Needed

Standard HuggingFace CausalLM models compute loss with hardcoded `reduction="mean"`:
```python
loss_fct = CrossEntropyLoss()  # Always averages the loss
```

Our method needs individual token losses to compute perplexity for each example:
```python
loss_fct = CrossEntropyLoss(reduction="none")  # Returns loss per token
```

### How to Apply the Patch

All training scripts automatically apply the patch. If you're using the models in your own code:

```python
from transformers import MistralForCausalLM
from models.patch_utils import patch_causal_lm_for_loss_reduction

# Load model
model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Apply patch (required for contrastive perplexity training)
model = patch_causal_lm_for_loss_reduction(model)

# Now you can use loss_reduction parameter
output = model(input_ids=..., labels=..., loss_reduction="none")
```

The patch utility is located at [`models/patch_utils.py`](models/patch_utils.py) and works with any HuggingFace CausalLM model (LLaMA, Mistral, GPT, etc.).

## Usage

### Data Preparation

The method requires paired data: neutral prompts and their toxic paraphrases. We use the ToxiGen dataset:

```bash
# Download and prepare ToxiGen data
bash scripts/prepare_data.sh

# Generate paraphrases using LLM (optional, pre-processed data available)
python scripts/paraphrase.py \
    --input_dir data/eval/toxigen \
    --output_dir data/paraphrase/toxigen \
    --model_name TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ
```

### Training

#### Contrastive Perplexity Training with Mistral

```bash
python train_mistral.py \
    --model_name mistralai/Mistral-7B-v0.1 \
    --data_dir data/paraphrase/safeNLP_processed \
    --output_dir results/mistral_contrastive \
    --num_pos 6 \
    --num_neg 6 \
    --alpha 100.0 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --use_4bit_quantization \
    --lora_r 64 \
    --lora_target_modules q_proj,v_proj,k_proj,o_proj \
    --bf16
```

#### Key Training Arguments

- `--num_pos`: Number of positive (non-toxic) paraphrases per batch
- `--num_neg`: Number of negative (toxic) paraphrases per batch  
- `--alpha`: Weight for the contrastive loss term (higher = stronger detoxification)
- `--tau`: Temperature parameter for contrastive learning
- `--smoothing`: Label smoothing factor for the contrastive loss

#### Training with Hard Negatives

For even stronger detoxification, use the hard negatives variant:

```bash
python train_mistral_hard_negatives.py \
    --model_name mistralai/Mistral-7B-v0.1 \
    --num_pos 5 \
    --num_neg 5 \
    --alpha 100.0 \
    --gradient_accumulation_steps 6 \
    --num_train_epochs 5 \
    --learning_rate 2.2e-5
```

### Alternative Training Methods

We also provide implementations of baseline methods for comparison:

```bash
# DPO (Direct Preference Optimization)
python train_dpo.py --model_name mistralai/Mistral-7B-v0.1

# PPO (Proximal Policy Optimization)  
python train_ppo.py --model_name mistralai/Mistral-7B-v0.1

# SimPO
python run_simpo.py --model_name mistralai/Mistral-7B-v0.1
```

### Evaluation

Compute perplexity on toxic vs. non-toxic text:

```bash
python scripts/compute_perplexity.py \
    --model_path results/mistral_contrastive \
    --data_dir data/paraphrase/safeNLP_processed \
    --num_toxic_samples 4 \
    --num_neutral_samples 0
```

Expected output: Lower perplexity on non-toxic text indicates successful detoxification.

## Repository Structure

```
.
├── models/
│   ├── patch_utils.py          # Utility to patch HuggingFace models for loss_reduction
│   └── README.md               # Documentation of model modifications
├── scripts/
│   ├── prepare_data.sh         # Download ToxiGen dataset
│   ├── paraphrase.py           # Generate paraphrases using LLMs
│   ├── clean_data.py           # Data cleaning and toxicity scoring
│   └── compute_perplexity.py   # Evaluation script
├── train_mistral.py            # Main training script (Mistral)
├── train_mistral_hard_negatives.py  # Hard negatives variant
├── train_mistral_cycle.py      # Cycle consistency variant
├── contrastive-train.py        # LLaMA-2 training
├── train_dpo.py                # DPO baseline
├── train_ppo.py                # PPO baseline
├── run_simpo.py                # SimPO baseline
└── requirements.txt            # Python dependencies
```

## Configuration

Training hyperparameters can be configured via:
1. Command-line arguments (see `--help` for each script)
2. YAML config files (in `configs/` directory)
3. Environment variables (for data paths, wandb settings)

Example config file:
```yaml
# configs/contrastive_mistral.yaml
model_name: mistralai/Mistral-7B-v0.1
num_pos: 6
num_neg: 6
alpha: 100.0
learning_rate: 2e-5
num_train_epochs: 3
```

Use with: `python train_mistral.py --config configs/contrastive_mistral.yaml`

## Results

Our method achieves:
- **67% reduction** in toxic generation (measured on RealToxicityPrompts)
- **Maintains 95%+ performance** on MMLU, HellaSwag, and other benchmarks
- **Lower perplexity** on non-toxic text compared to baseline methods

See the [paper](https://arxiv.org/abs/2401.08491) for full experimental results and analysis.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{klein2025contrastive,
  title={Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models},
  author={Klein, Tassilo and Nabi, Moin},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)},
  year={2025}
}
```

## How to obtain support
[Create an issue](https://github.com/SAP-samples/acl2025-contrastive-perplexity/issues) in this repository if you find a bug or have questions about the content.
 
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2025 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.
