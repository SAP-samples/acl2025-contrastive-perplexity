"""
Contrastive Perplexity Training for LLaMA-2 Models

LLaMA-2 version of the contrastive perplexity training method.
See train_mistral.py for the main implementation with Mistral models.

This script follows the same methodology but is configured for LLaMA-2 models.

Usage:
    python contrastive-train.py \\
        --model_name meta-llama/Llama-2-7b-hf \\
        --data_dir data/paraphrase/safeNLP_processed \\
        --output_dir results/llama2_contrastive

Authors: Tassilo Klein, Moin Nabi
License: Apache 2.0
"""
# %%

import argparse
import glob
import einops
import json
import sys
import os
import random
from collections import defaultdict
from tqdm import tqdm, trange
from datasets import Dataset
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import transformers
from dataclasses import dataclass, field
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import TrainerCallback, GenerationConfig
from transformers.utils import PaddingStrategy
from datasets import load_dataset
import wandb
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

# %%
base_model = "meta-llama/Llama-2-7b-hf"
device_map = "auto"
# %%
num_epochs = 3
val_set_size = 0
eval_steps = 100
micro_batch_size = 4
gradient_accumulation_steps = 4
warmup_steps = 500
learning_rate = 1e-4
output_dir = "results/"
group_by_length = False
use_wandb = False
ddp = False
lora_r= 8
lora_alpha = 16
lora_dropout = 0.05
lora_target_modules = [
    "q_proj",
    "v_proj",
]
wandb_project = "GenDetox"
wandb_run_name = ""

wandb.init(project=wandb_project)

if not(wandb.run.name is None):
        output_name = wandb.run.name
else:
    output_name = 'dummy-run'

output_dir = os.path.join(
    output_dir, output_name)

# Check if parameter passed or if set within environ
use_wandb = len(wandb_project) > 0 or (
    "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
)

resume_from_checkpoint = None

# %%
from transformers import LlamaTokenizer, LlamaConfig, AutoModelForCausalLM, LlamaModel, LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)#
config = LlamaConfig.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=quantization_config,
    )

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)


model = get_peft_model(model, config)

tokenizer = LlamaTokenizer.from_pretrained(base_model)


tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"  # Allow batched inference
cutoff_len = 256

def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        #result["labels"] = result["input_ids"].copy()
        
        return result


# %%
random.seed(42)

# If you want to include system prompt, see this discussion for the template: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/discussions/4
# However, see here that removing the system prompt actually reduce the false refusal rates: https://github.com/facebookresearch/llama/blob/main/UPDATES.md?utm_source=twitter&utm_medium=organic_social&utm_campaign=llama2&utm_content=text#observed-issue
llama2_prompting_template = '''[INST] {prompt} [/INST]'''

# See the tulu template here: https://huggingface.co/allenai/tulu-7b#input-format
tulu_prompting_template = '''<|user|>\n{prompt}\n<|assistant|>\n'''

class Args:
    pass

args = Args()

args.data_dir = "/home/ubuntu/open-instruct/data/paraphrase/safeNLP_processed"
args.max_prompts_per_group = 500
args.use_llama_chat_format = False
args.use_chat_format = False
args.base_model = base_model

#tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
pos_prompts = dict()
neg_prompts = dict()

groups = ['asian', 'black', 'chinese', 'jewish', 'latino', 'lgbtq', 'mental_dis', 'mexican', 'middle-eastern', 'muslim', 'native-american', 'physical_dis', 'women']
groups = [f"neutral_{group}.json" for group in groups]
dataset = load_dataset('json', data_dir=args.data_dir,data_files={'train': groups,}) 


for data in tqdm(dataset['train']):
    
    for curr_dict, tag in zip([pos_prompts, neg_prompts],["paraphrases","paraphrases_toxic"]):
        prompt_list = []
        for prompt in data[tag]:
            prompt = prompt.replace("\\\\", "\\")
            prompt = prompt.replace("\\n", "\n")
            
            if args.use_llama_chat_format:
                prompt = llama2_prompting_template.format(
                    prompt="Complete the following: " + prompt
                )
            elif args.use_chat_format:
                prompt = (
                    "<|user|>\nComplete the following: "
                    + prompt
                    + "\n<|assistant|>\nA:"
                )
            else:
                prompt = prompt
                
            prompt = tokenize(prompt)
            prompt_list.append(prompt)
        curr_dict[data['id']] = prompt_list

# %%
def generate_and_tokenize_prompt_pairs(data_point, **kwargs):
    
    args = kwargs['args']
    
    if args.use_llama_chat_format:
        prompt = llama2_prompting_template.format(
            prompt="Complete the following: " + data_point["instruction"]
        )
    elif args.use_chat_format:
        prompt = (
            "<|user|>\nComplete the following: "
            + data_point["instruction"]
            + "\n<|assistant|>\nA:"
        )
    else:
        prompt = data_point["input"]
        
    prompt = tokenize(prompt)
    #prompt["labels"] = prompt["input_ids"].copy()
    
    prompt["label"] = data_point["id"]
    return prompt



@dataclass
class MyDataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
            sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
            acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    pos_dict: dict
    neg_dict: dict
    num_pos: int
    num_hard_negs: int
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, pre_features, return_tensors=None):
        features = []
        if return_tensors is None:
            return_tensors = self.return_tensors
            
        
                
        for feature_idx, feature in enumerate(pre_features):
            # positive
            features.append({"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"]})
            
            for nums, curr_dict in zip([self.num_pos, self.num_hard_negs], [self.pos_dict, self.neg_dict]):
                curr_pool = np.arange(len(curr_dict[feature["label"]]))
                np.random.shuffle(curr_pool)
                for i in range(nums):
                    item = curr_dict[feature["label"]][curr_pool[i]]
                    features.append({"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]})
        
        features = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
        return features

class CustomTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        
        tmp = 1
        var_lambda = 0.1
        
        output = model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],labels=inputs['input_ids'], return_dict=True, loss_reduction="none")
        loss = einops.rearrange(output['loss'], " (B S) -> B S", B=inputs['input_ids'].shape[0])
        loss = torch.nan_to_num(loss, nan=5.0, posinf=5.0)
        if True:
            loss_fct = nn.CrossEntropyLoss()
            
            
            NUM_POS = 2 #
            NUM_SAMPLES = NUM_POS + 1 # + reference
            NUM_NEG =  6# must be dividible by 2
            NUM_RAND_NEG = 0
            NUM_ELEMENTS = NUM_SAMPLES+NUM_NEG+NUM_RAND_NEG
            
            true_micro_batch_size = int(inputs['input_ids'].shape[0]/(NUM_SAMPLES+NUM_NEG+NUM_RAND_NEG))
            
            index_tensor = np.arange(true_micro_batch_size*NUM_ELEMENTS).reshape((true_micro_batch_size, NUM_ELEMENTS))
    
            firstLevel_pos_idx = index_tensor[:,1:NUM_SAMPLES].flatten()
            firstLevel_neg_idx = index_tensor[:,NUM_SAMPLES:].flatten()
            firstLevel_ref_idx = index_tensor[:,:1].flatten()

            
            firstLevel_ref_idx_reshaped_pos = einops.repeat(firstLevel_ref_idx,"h -> (h p)", p=NUM_POS)
            firstLevel_ref_idx_reshaped_neg = einops.repeat(firstLevel_ref_idx,"h -> (h p)", p=NUM_NEG)
            
            input_scale = torch.sum(inputs['attention_mask']==1,dim=1)/torch.max(torch.sum(inputs['attention_mask']==1,dim=1))

            

            
            pos_scores_ = -torch.abs(torch.mean(loss[firstLevel_pos_idx]*input_scale[firstLevel_pos_idx].unsqueeze(1),dim=1) - torch.mean(loss[firstLevel_ref_idx_reshaped_pos]*input_scale[firstLevel_ref_idx_reshaped_pos].unsqueeze(1),dim=1)) / 1.0
            neg_scores_ = -torch.abs(torch.mean(loss[firstLevel_neg_idx]*input_scale[firstLevel_neg_idx].unsqueeze(1),dim=1) - torch.mean(loss[firstLevel_ref_idx_reshaped_neg]*input_scale[firstLevel_ref_idx_reshaped_neg].unsqueeze(1),dim=1)) / 1.0
            
            
            neg_scores_ = einops.rearrange(neg_scores_,"(a b)-> a b",a=micro_batch_size)
            pos_scores_ = einops.rearrange(pos_scores_,"(a b)-> a b",a=micro_batch_size)
            final_scores = torch.cat([torch.pow(2.,pos_scores_), torch.pow(2.0, neg_scores_)], 1)
            label_probs_ = torch.cat([torch.ones_like(pos_scores_), torch.zeros_like(neg_scores_)],1).to(final_scores.device)
            constrastive_loss_perperplexity = loss_fct(final_scores, label_probs_)
            
            wandb.log({'train/perplexity_loss': constrastive_loss_perperplexity.item()})
            
            return constrastive_loss_perperplexity
        

# %%
trainer = CustomTrainer(
        model=model,
        train_dataset=train_data['train'],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if val_set_size > 0 else None,
            save_steps=eval_steps,
            output_dir=output_dir,
            save_total_limit=6,
            dataloader_num_workers=10,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            dataloader_drop_last=True,
        ),
        data_collator=MyDataCollatorForSeq2Seq(
            tokenizer, pos_prompts, neg_prompts, num_pos=2,num_hard_negs=6,pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

# %%
if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

trainer._signature_columns=["input_ids", "attention_mask", "label",]
# %%
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

model.save_pretrained(output_dir)


