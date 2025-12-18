"""
DPO (Direct Preference Optimization) Baseline Training Script

Implements Direct Preference Optimization as a baseline method for comparison
with contrastive perplexity.

DPO directly optimizes the policy to prefer non-toxic over toxic completions
without requiring a separate reward model.

Usage:
    python train_dpo.py --model_name mistralai/Mistral-7B-v0.1

Authors: Tassilo Klein, Moin Nabi
License: Apache 2.0
"""

#
# SPDX-FileCopyrightText: 2025 SAP SE or an SAP affiliate company
#
# SPDX-License-Identifier: Apache-2.0
#

import transformers
from dataclasses import dataclass, field
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import TrainerCallback, GenerationConfig
from transformers.utils import PaddingStrategy
from datasets import load_dataset
import wandb
from random import choice
import random
import os
import torch
from trl import DPOTrainer
import sys
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,

    set_peft_model_state_dict,
)
from unsloth import FastLanguageModel
from transformers import LlamaTokenizer, LlamaConfig, MistralConfig, AutoModelForCausalLM, LlamaModel, LlamaForCausalLM, MistralForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from datasets import load_dataset
train_dataset = load_dataset('json', data_files={'train':'dpo_data.json'})

def return_prompt_and_responses(samples):
    return {
    "prompt": [
    f"### Input: ```{input}```\n ### Output: "
    for input in samples["input"]
    ],
    "chosen": samples["chosen"],
    "rejected": samples["rejected"],
    }
    
train_dataset = load_dataset("json", data_files="dpo_data.json",split="train")
original_columns = train_dataset.column_names
train_dataset = train_dataset.map(
 return_prompt_and_responses,
 batched=True,
 remove_columns=original_columns
)

class Args:
    pass

args = Args()
args.description = "detox-DPO"
args.tags = None
args.use_4bit_quantization = False
args.use_8bit_quantization = False
args.output_dir = "results"
args.model_name = "mistralai/Mistral-7B-v0.1"
args.bnb_4bit_quant_type = "nf4"
args.bnb_4bit_compute_dtype = "float16"
args.use_nested_quant = True
args.lora_target_modules = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj,lm_head"
args.per_device_train_batch_size = 2 #2
args.gradient_accumulation_steps = 3 #3
args.num_train_epochs = 1
args.bf16 = True
args.lora_r = 64
args.weight_decay = 0.0
args.lr_scheduler_type = "linear"
args.eval_steps = None
args.save_steps = 10
args.warmup_steps = 0
args.warmup_ratio = 0.000
args.use_gradient_checkpointing = False
args.optim = "adamw_torch"
args.num_workers = 4
args.beta=0.1
args.push_to_hub = False
args.lora_alpha = 16
args.lora_dropout = 0.1
args.seed = 42
args.learning_rate = 2e-4
args.fp16 = False
args.logging_steps = 10
#-num_pos 3 --num_neg 7 


wandb_project = args.description #_hard_negatives"
wandb_run_name = ""

if not (args.tags is None):
    args.tags = [item for item in args.tags.split(',')]

if not(args.tags is None):
    wandb.init(project=args.description, tags=args.tags)

else:
    wandb.init(project=args.description)

if not(wandb.run.name is None):
        output_name = wandb.run.name
else:
    output_name = 'dummy-run'

args.output_dir = os.path.join(
    args.output_dir, output_name)

# Check if parameter passed or if set within environ
use_wandb = len(wandb_project) > 0 or (
    "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
)

resume_from_checkpoint = None
if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
                
elif args.use_8bit_quantization:
    compute_dtype = getattr(torch, args.bnb_8bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=args.use_8bit_quantization,
        bnb_8bit_quant_type=args.bnb_8bit_quant_type,
        bnb_8bit_compute_dtype=compute_dtype,
        bnb_8bit_use_double_quant=args.use_nested_quant,
    )

    if compute_dtype == torch.float16 and args.use_4bit_quantization:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)
else:
    pass

device_map = "auto"
    
config = MistralConfig.from_pretrained(args.model_name)


max_seq_length = 2048 # Supports automatic RoPE Scaling, so choose any number

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_name,
    max_seq_length = max_seq_length,
    dtype=None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
    load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
)


# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = args.lora_r,
    target_modules = [item for item in args.lora_target_modules.split(',')],
    lora_alpha = args.lora_alpha,
    lora_dropout = args.lora_dropout, # Dropout = 0 is currently optimized
    bias = "none",    # Bias = "none" is currently optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
)




tokenizer = AutoTokenizer.from_pretrained(args.model_name)

tokenizer.pad_token_id = (
    tokenizer.eos_token_id  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"  # Allow batched inference
cutoff_len = 256

# %%
random.seed(args.seed)


dpo_trainer = DPOTrainer(
    model,
    max_length=512,
    max_prompt_length=128,
    ref_model=None,
    beta=args.beta,
    args=transformers.TrainingArguments(
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                gradient_checkpointing=args.use_gradient_checkpointing,
                warmup_steps=args.warmup_steps,
                warmup_ratio=args.warmup_ratio,
                num_train_epochs=args.num_train_epochs,
                learning_rate=args.learning_rate,
                fp16=args.fp16,
                bf16=args.bf16,
                logging_steps=args.logging_steps,
                evaluation_strategy="no",
                save_strategy="no",
                eval_steps=args.eval_steps,
                save_steps=args.save_steps,
                save_total_limit=0,
                dataloader_num_workers=args.num_workers,
                load_best_model_at_end=False,
                ddp_find_unused_parameters=False,
                group_by_length=False,
                run_name=wandb_run_name if use_wandb else None,
                dataloader_drop_last=True,
                output_dir=args.output_dir,
                optim=args.optim,
                push_to_hub=args.push_to_hub,
                report_to="wandb",
                ),
    train_dataset=train_dataset,
    tokenizer=tokenizer
    )

dpo_trainer.train()
dpo_trainer.save_model(args.output_dir)
