"""
Contrastive Perplexity Training for Mistral Models

This script implements the main contrastive perplexity training method for detoxifying
large language models, as described in the ACL 2025 paper:
"Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models"

The method trains models to minimize perplexity on non-toxic text while maximizing it on
toxic text, using a contrastive objective with positive (non-toxic) and negative (toxic) paraphrases.

Key Features:
- Supports 4-bit quantization with QLoRA for memory efficiency
- Contrastive perplexity loss with configurable alpha weighting
- Automatic model patching for per-token loss computation
- Integration with Weights & Biases for experiment tracking

Usage:
    python train_mistral.py \\
        --model_name mistralai/Mistral-7B-v0.1 \\
        --data_dir data/paraphrase/safeNLP_processed \\
        --output_dir results/mistral_contrastive \\
        --num_pos 6 --num_neg 6 --alpha 100.0

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
from transformers import HfArgumentParser, TrainingArguments, Trainer
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
from random import choice

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaTokenizer, LlamaConfig, MistralConfig, AutoModelForCausalLM, LlamaModel, LlamaForCausalLM, MistralForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    alpha: Optional[float] = field(default=100.0)
    beta: Optional[float] = field(default=1.0)
    detox: Optional[float] = field(default=1.0)
    toxify: Optional[float] = field(default=1.0)
    tau: Optional[float] = field(default=1.0)
    smoothing: Optional[float] = field(default=0.0)
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    num_pos: Optional[int] = field(default=6)
    num_neg: Optional[int] = field(default=6)
    num_rnd: Optional[int] = field(default=6)
    seed: Optional[int] = field(default=42)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[float] = field(default=0.001),
    max_prompts_per_group: Optional[int] = field(default=500),
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="mistralai/Mistral-7B-Instruct-v0.1",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    use_instruction: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use instruction tuning"},
    )
    use_llama_chat_format: Optional[bool] = field(
        default=False,
        metadata={"help": "Llama chat format"},
    )
    use_chat_format: Optional[bool] = field(
        default=False,
        metadata={"help": "Chat format"},
    )
    use_8bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    bnb_8bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_8bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=4,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=10000, metadata={"help": "How many optimizer update steps to take"})
    warmup_steps: int = field(default=0, metadata={"help": "Number of steps to do a warmup for"})
    warmup_ratio: float = field(default=0.0, metadata={"help": "Number of steps to do a warmup for"})
    save_steps: int = field(default=10, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Eval model every X steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(default="results", metadata={"help": "Where to store the final model."})
    data_dir: str = field(default="/home/ubuntu/open-instruct/data/paraphrase/safeNLP_processed", metadata={"help": "Where training data is stored."})
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Gradient Checkpointing."},
    )
    dataset_text_field: str = field(default="text", metadata={"help": "Dataset field to use as input text."})
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, pushes the model to the HF Hub"},
    )
    num_workers: int = field(default=4, metadata={"help": "Number of dataset workers to use."})
    debug: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tests things like proper saving/loading/logging of model"},
    )
    description: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb experiment description"})
    tags:  Optional[str] = field(default=None,
                                 metadata={"help": 'tags for wandb, comma separated '})

def main(args):
    """Main training function for contrastive perplexity method.
    
    Sets up model, data, and trainer for contrastive perplexity training where the model
    learns to distinguish between non-toxic (positive) and toxic (negative) text by
    minimizing perplexity on positive samples while maximizing it on negatives.
    
    Args:
        args: ScriptArguments dataclass containing all training configuration including:
            - model_name: HuggingFace model identifier
            - data_dir: Path to paraphrase dataset directory
            - num_pos: Number of positive (non-toxic) samples per batch
            - num_neg: Number of negative (toxic) samples per batch
            - num_rnd: Number of random negative samples per batch
            - alpha: Weight for contrastive loss component
            - Other hyperparameters for training and quantization
    """
    # Create index vectors to identify positive (2), negative (3), and random negative (4) samples
    # in the batched data. This allows efficient indexing during loss computation.
    idx_vec = []
    for i in range(args.per_device_train_batch_size):
        idx_vec.append(torch.cat((
            2*torch.ones(int(args.num_pos), 1),     # Positive samples
            3*torch.ones(args.num_neg, 1),           # Hard negative samples  
            4*torch.ones(args.num_rnd, 1)            # Random negative samples
        ), dim=0))
        
    idx_vec = torch.cat(idx_vec)
    seq_pos_idx = torch.where(idx_vec == 2)[0]  # Indices of positive samples
    seq_neg_idx = torch.where(idx_vec == 3)[0]  # Indices of hard negative samples
    seq_rnd_idx = torch.where(idx_vec == 4)[0]  # Indices of random negative samples  
    


    wandb_project = args.description
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

        
    device_map = "auto"
        
    config = MistralConfig.from_pretrained(args.model_name)
    model = MistralForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map=device_map,
            quantization_config=bnb_config,
            use_flash_attention_2=True
        )

    model = prepare_model_for_int8_training(model)

    if len(args.lora_target_modules) > 0:
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[item for item in args.lora_target_modules.split(',')],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
            config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )


    model = get_peft_model(model, config)


    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenizer.pad_token_id = (
        tokenizer.eos_token_id  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    cutoff_len = 256

    def tokenize(prompt, add_eos_token=True):
        """Tokenize text prompt with truncation and optional EOS token.
        
        Args:
            prompt: Text string to tokenize
            add_eos_token: If True, append EOS token if not present and sequence is under max length
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' keys containing tokenized sequences
        """
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        # Ensure EOS token is present for proper sequence termination
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)


            
            return result


    # %%
    random.seed(args.seed)

    # If you want to include system prompt, see this discussion for the template: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/discussions/4
    # However, see here that removing the system prompt actually reduce the false refusal rates: https://github.com/facebookresearch/llama/blob/main/UPDATES.md?utm_source=twitter&utm_medium=organic_social&utm_campaign=llama2&utm_content=text#observed-issue
    llama2_prompting_template = '''[INST] {prompt} [/INST]'''

    # See the tulu template here: https://huggingface.co/allenai/tulu-7b#input-format
    tulu_prompting_template = '''<|user|>\n{prompt}\n<|assistant|>\n'''

    class Args:
        pass




    pos_prompts = dict()
    neg_prompts = dict()
    rnd_neg_prompts = dict()

    groups = ['asian', 'black', 'chinese', 'jewish', 'latino', 'lgbtq', 'mental_dis', 'mexican', 'middle-eastern', 'muslim', 'native-american', 'physical_dis', 'women']
    groups = [f"neutral_{group}.json" for group in groups]
    dataset = load_dataset('json', data_dir=args.data_dir,data_files={'train': groups,}) 

    remove_id = []

    group_set = set([x['id'] for x in dataset['train']])
    for i in group_set:
        rnd_neg_prompts[i] = []

    for idx,data in enumerate(tqdm(dataset['train'])):
        
        without_current_group = [x for x in group_set if not(x.startswith(data['id'][:data['id'].find("-")]))]
        
        for num, curr_dict, tag in zip([int(args.num_pos), args.num_neg],[pos_prompts, neg_prompts],["paraphrases","paraphrases_toxic"]):
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
                    

                prompt_list.append(prompt)
                
                if tag == "paraphrases_toxic":
                    NUM_RAND_SAMPLES = 200 if args.num_rnd > 0 else 0
                    for sample in range(NUM_RAND_SAMPLES):
                        rnd_neg= choice(without_current_group)
                        rnd_neg_prompts[rnd_neg] = rnd_neg_prompts[rnd_neg] + [prompt]
            if data['id'] == 'middle-eastern-65':
                tmp = 1
                
                
            curr_dict[data['id']] = prompt_list
            if len(curr_dict[data['id']]) < num:
                remove_id.append(idx)
                
    dataset['train'] = dataset['train'].select(
        (
            i for i in range(len( dataset['train'])) 
            if i not in set(remove_id)
        )
    )

    print(f"Number of removed items: {len(np.unique(remove_id))}")
    
    def generate_and_tokenize_prompt_pairs(data_point, **kwargs):
        \"\"\"Process dataset items to extract reference text and attach group labels.
        
        Formats prompts based on specified chat template (LLaMA-2, Tulu, or plain text)
        and attaches the identity group label for paraphrase matching during training.
        
        Args:
            data_point: Dictionary containing 'input', 'instruction', and 'id' keys
            **kwargs: Must contain 'args' with use_llama_chat_format and use_chat_format flags
            
        Returns:
            Dictionary with 'input_ids' (text prompt) and 'label' (group ID) keys
        \"\"\"
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
            
        prompt = {'input_ids': prompt}
        
        prompt["label"] = data_point["id"]
        return prompt
    train_data = dataset.shuffle().map(generate_and_tokenize_prompt_pairs,fn_kwargs={"args":args}).remove_columns(["input", "id", "paraphrases", "paraphrases_toxic","group"])

    @dataclass
    class MyDataCollatorForSeq2Seq:
        """Data collator for contrastive perplexity training with positive and negative paraphrase sampling.
        
        This custom collator creates training batches by sampling toxic and non-toxic paraphrases
        of the same reference text. For each batch item, it includes:
        - Positive samples: Non-toxic paraphrases from the same identity group
        - Hard negative samples: Toxic paraphrases from the same identity group
       - Random negative samples: Toxic paraphrases from different identity groups
        
        This enables contrastive learning where the model learns to assign high perplexity to
        toxic text and low perplexity to non-toxic text.

        Args:
            tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
                The tokenizer used for encoding the data.
            pos_dict: Dictionary mapping group IDs to lists of non-toxic paraphrases
            neg_dict: Dictionary mapping group IDs to lists of toxic paraphrases
            rnd_dict: Dictionary mapping group IDs to random toxic samples from other groups
            num_pos (int): Number of positive (non-toxic) samples per batch item
            num_hard_negs (int): Number of hard negative (toxic) samples from same group
            num_rnd_negs (int): Number of random negative samples from different groups
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences among:
                - `True` or `'longest'` (default): Pad to the longest sequence in the batch
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length`
                - `'do_not_pad'`: No padding (output batch with sequences of different lengths)
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. Useful for Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            label_pad_token_id (`int`, *optional*, defaults to -100):
                The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
            return_tensors (`str`):
                The type of Tensor to return. Allowable values are "np", "pt" and "tf".
        """

        tokenizer: PreTrainedTokenizerBase
        pos_dict: dict
        neg_dict: dict
        rnd_dict: dict
        num_pos: int
        num_hard_negs: int
        num_rnd_negs: int
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        label_pad_token_id: int = -100
        return_tensors: str = "pt"

        def __call__(self, pre_features, return_tensors=None):
            features = []
            if return_tensors is None:
                return_tensors = self.return_tensors
            

            proxy_target_len = []
            for feature_idx, feature in enumerate(pre_features):
                
                # positive

                
                # circular
                pos_features = []
                neg_features = []
                rnd_neg_features = []
                # add reference so it may be used
                _pos_list = self.pos_dict[feature["label"]] + [feature['input_ids']]
                pos_pool = np.arange(len(_pos_list))   
                for cls, nums, curr_dict in zip(["POS", "NEG", "RND_NEG"],[int(self.num_pos), self.num_hard_negs, int(self.num_rnd_negs)], [self.pos_dict, self.neg_dict, self.rnd_dict]):
                    curr_pool = np.arange(len(curr_dict[feature["label"]]))
                    
                    np.random.shuffle(pos_pool)
                    
                    np.random.shuffle(curr_pool)
                    for i in range(nums):
                        item = curr_dict[feature["label"]][curr_pool[i]]
                    
                        if cls == "POS": 
                            pos_features.append(tokenize(item))

                        if cls == "RND_NEG": # translate NEG to REF
                            
                            rnd_neg_features.append(tokenize(item))
                        if cls == "NEG":
                            #translate from REF to NEG
                            #prompt = f'<|user|>\nParaphrase the following sentence in a toxic and possibly racist way: {pos_item} </s>\n<|assistant|>\n{item}'
                            
                            if args.use_instruction:
                                prompt = f'<s>[INST] Paraphrase the following sentence in a toxic and possibly racist way: {pos_item} [\INST] {item}<\s>'
                            else:
                                prompt = item
                            proxy_target = tokenize(prompt)
                            neg_features.append(tokenize(item))
                features = features + pos_features + neg_features + rnd_neg_features

            features = self.tokenizer.pad(
                    features,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=return_tensors,
                )
            return features

    class CustomTrainer(transformers.Trainer):
        \"\"\"Custom trainer implementing contrastive perplexity loss for detoxification.
        
        Computes a custom loss that encourages low perplexity on positive (non-toxic) samples
        and high perplexity on negative (toxic) samples. The loss uses per-token perplexity
        differences weighted by temperature and alpha parameters.
        \"\"\"
        
        def __init__(self, num_pos, num_neg, num_rnd, batch_size, alpha, beta, detox, toxify, tau, smoothing, *args, **kwargs):
            \"\"\"Initialize custom trainer with contrastive loss parameters.
            
            Args:
                num_pos: Number of positive (non-toxic) samples per batch
                num_neg: Number of hard negative (toxic) samples per batch
                num_rnd: Number of random negative samples per batch
                batch_size: Micro batch size for training
                alpha: Weight for toxic sample penalty in contrastive loss
                beta: Overall weight for contrastive perplexity loss
                detox: Weight for detoxification objective
               toxify: Weight for toxification penalty
                tau: Temperature parameter for softmax scaling
                smoothing: Label smoothing factor for cross-entropy loss
                *args, **kwargs: Additional arguments passed to base Trainer
            \"\"\"
            super().__init__(*args, **kwargs)
            self.num_pos = num_pos
            self.num_neg = num_neg
            self.num_rnd = num_rnd
            self.batch_size = batch_size
            self.alpha = alpha
            self.beta = beta
            self.detox = detox
            self.toxify = toxify
            self.tau = tau
            self.smoothing = smoothing
        def compute_loss(self, model, inputs, return_outputs=False):
            """Compute contrastive perplexity loss for detoxification.            
            The core contrastive objective: minimize perplexity on non-toxic (positive) samples
            while maximizing it on toxic (negative) samples. This is achieved by:
            1. Computing per-token cross-entropy loss for all samples
            2. Calculating mean perplexity for positive samples as reference
            3. Computing perplexity-based scores where samples closer to positive reference score higher
            4. Applying weighted cross-entropy to prefer positive samples over negatives
            
            Args:
                model: The language model being trained
                inputs: Dictionary with 'input_ids' and 'attention_mask' keys
                return_outputs: If True, return model outputs in addition to loss
                
            Returns:
                Weighted contrastive perplexity loss (scalar tensor)
            """
            # Forward pass with per-token loss (loss_reduction="none" gives token-level losses)
            proxy_inputs = inputs['input_ids']
            proxy_masks = inputs['attention_mask']
            proxy_labels = inputs['input_ids']  # Self-supervised: predict next token
            output = model(input_ids=proxy_inputs, attention_mask=proxy_masks, labels=proxy_labels, 
                         return_dict=True, loss_reduction="none")
            
            # Reshape loss from (B*S,) to (B, S) where B=batch_size, S=seq_len
            loss = einops.rearrange(output['loss'], " (B S) -> B S", B=inputs['input_ids'].shape[0])
            loss = torch.nan_to_num(loss, nan=5.0, posinf=5.0)  # Handle numerical instabilities
            
            # Constants defining batch composition
            NUM_POS = args.num_pos  # Number of positive (non-toxic) samples
            NUM_NEG = args.num_neg  # Number of hard negative (toxic) samples from same group
            NUM_RAND_NEG = args.num_rnd  # Number of random negative samples from other groups
            NUM_ELEMENTS = NUM_POS + NUM_NEG + NUM_RAND_NEG
            micro_batch_size = self.batch_size
            true_micro_batch_size = int(inputs['input_ids'].shape[0] / NUM_ELEMENTS)
            
            # Create index arrays to select positive/negative samples from the batch
            # seq_pos_idx, seq_neg_idx, seq_rnd_idx were precomputed in main()
            index_tensor = np.arange(true_micro_batch_size * NUM_ELEMENTS)
            firstLevel_pos_idx = index_tensor[seq_pos_idx].flatten()  # Indices of positive samples
            firstLevel_neg_idx = index_tensor[seq_neg_idx].flatten()  # Indices of hard negatives
            firstLevel_rnd_idx = index_tensor[seq_rnd_idx].flatten()  # Indices of random negatives

            # Compute reference perplexity as average over all positive samples
            # This serves as the "target" perplexity for non-toxic text
            ref_pos_avg = torch.mean(torch.mean(loss[firstLevel_pos_idx], dim=1))
            
            # Compute contrastive scores: negative absolute distance from reference
            # Samples with perplexity similar to positive reference get higher (less negative) scores
            pos_scores_ = -torch.abs(torch.mean(loss[firstLevel_pos_idx], dim=1) - ref_pos_avg)
            neg_scores_ = -torch.abs(torch.mean(loss[firstLevel_neg_idx], dim=1) - ref_pos_avg)
            rnd_scores_ = -torch.abs(torch.mean(loss[firstLevel_rnd_idx], dim=1) - ref_pos_avg)
            
            # Reshape scores from flat to (micro_batch_size, num_samples_per_item)
            # This groups scores by their corresponding reference text
            neg_scores_ = einops.rearrange(neg_scores_, "(a b)-> a b", a=micro_batch_size)
            rnd_scores_ = einops.rearrange(rnd_scores_, "(a b)-> a b", a=micro_batch_size)
            pos_scores_ = einops.rearrange(pos_scores_, "(a b)-> a b", a=micro_batch_size)
            
            # Apply temperature scaling via exp() and tau, then concatenate all scores
            # Temperature (tau) controls how confident the model is about distinctions
            loss_fct = CrossEntropyLoss(label_smoothing=args.smoothing)
            final_scores = torch.cat([
                torch.exp(pos_scores_),  # Positive samples should have high probability
                torch.exp(neg_scores_),  # Hard negatives
                torch.exp(rnd_scores_)   # Random negatives
            ], dim=1) / self.tau
            
            # Binary labels: 1 for positive (non-toxic), 0 for negative (toxic)
            label_probs_ = torch.cat([
                torch.ones_like(pos_scores_),
                torch.zeros_like(neg_scores_),
                torch.zeros_like(rnd_scores_)
            ], dim=1).to(final_scores.device)
            
            # Apply additional penalty (alpha) to hard negative samples
            # This increases the cost of assigning high probability to toxic text
            weights = torch.cat([
                torch.zeros_like(pos_scores_),
                self.alpha * torch.ones_like(neg_scores_),  # Penalize hard negatives more
                torch.zeros_like(rnd_scores_)
            ], dim=1).to(final_scores.device)
            
            final_scores = final_scores + weights
            
            # Compute cross-entropy loss between predicted scores and binary labels
            # Normalizing by number of samples to make loss magnitude independent of batch size
            contrastive_loss_perperplexity = loss_fct(final_scores, label_probs_) / (NUM_POS + 1 + NUM_RAND_NEG)
            
            wandb.log({'train/perplexity_loss': contrastive_loss_perperplexity.item()})
            
            # Scale final loss by beta hyperparameter
            return self.beta * contrastive_loss_perperplexity
            

    # %%
    trainer = CustomTrainer(
            model=model,
            train_dataset=train_data['train'],
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
                save_strategy="steps",
                eval_steps=args.eval_steps,
                save_steps=args.save_steps,
                save_total_limit=6,
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
            data_collator=MyDataCollatorForSeq2Seq(
                    tokenizer, pos_prompts, neg_prompts, rnd_neg_prompts, num_pos=args.num_pos,num_hard_negs=args.num_neg,num_rnd_negs=args.num_rnd,pad_to_multiple_of=8, return_tensors="pt", padding=True),
               
            num_pos=args.num_pos,
            num_neg=args.num_neg,
            num_rnd=args.num_rnd,
            batch_size=args.per_device_train_batch_size,
            alpha=args.alpha,
            beta=args.beta,
            detox=args.detox,
            toxify=args.toxify,
            tau=args.tau,
            smoothing=args.smoothing,
            )
    if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    trainer._signature_columns=["input_ids", "attention_mask", "label",]
    # %%
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
