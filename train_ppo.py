# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
"""
PPO (Proximal Policy Optimization) Baseline Training Script

Implements Proximal Policy Optimization with a toxicity reward model as a baseline
method for comparison with contrastive perplexity.

PPO uses reinforcement learning to optimize the policy based on a learned reward model
that scores text for toxicity.

Usage:
    python train_ppo.py --model_name mistralai/Mistral-7B-v0.1

Authors: Tassilo Klein, Moin Nabi
License: Apache 2.0
"""
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
import wandb
import trl

from transformers import MistralConfig,MistralForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPTJ model to generate less toxic contents
# by using allenai/real-toxicity-prompts dataset. We use PPO
#  (proximal policy optimization) to optimize the model.
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `project_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    model_name: Optional[str] = field(default="mistralai/Mistral-7B-v0.1", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "the number of gradient accumulation steps"}
    )
    model_save_path: Optional[str] = field(
        default="./mistral-detox-ppo-final",
        metadata={"help": "the path to save the model"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=1,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    config, dataset_name="allenai/real-toxicity-prompts", input_min_text_length=5, input_max_text_length=1000
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files="dpo_data.json",split="train")



    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        prompt = sample["prompt"]
        continuation = sample["rejected"]

        sample["input_ids"] = tokenizer.encode(prompt + continuation)[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]

    return ds


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_quant_type="nf4",
#bnb_4bit_compute_dtype="float16", 
bnb_4bit_compute_dtype=torch.bfloat16,
bnb_4bit_use_double_quant=False,)

mistral_config = MistralConfig.from_pretrained(config.model_name)
model = MistralForCausalLM.from_pretrained(
        config.model_name,
        config=mistral_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        quantization_config=bnb_config, attn_implementation="flash_attention_2")


    #     use_flash_attention_2=True

    
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
args.seed = 35
args.learning_rate = 1e-5
args.fp16 = False
args.logging_steps = 10


    

model_kwargs = None




lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=[item for item in args.lora_target_modules.split(',')],
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

model = AutoModelForCausalLMWithValueHead.from_pretrained(model)



# We create a reference model by sharing 20 layers
ref_model = create_reference_model(model, num_shared_layers=20)

# We make sure to use `Adam` optimizer on the model parameters that require gradients.
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

# GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# We retrieve the dataloader by calling the `build_dataset` function.
min_input_length = 30
max_input_length = 40
dataset = build_dataset(config, input_min_text_length=min_input_length, input_max_text_length=max_input_length)

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the reward pipeline, we will use the toxicity model to compute the reward.
# We first load the toxicity model and tokenizer.
toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
# We load the toxicity model in fp16 to save memory.
toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float16).to(
    ppo_trainer.accelerator.device
)


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
output_min_length = 20
output_max_length = 30
output_length_sampler = LengthSampler(output_min_length, output_max_length)

model_save_path = script_args.model_save_path

wandb.init(project="trl-ppo")

if not(wandb.run.name is None):
    model_save_path = "./results/"+wandb.run.name



for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from the policy model
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute sentiment score
    texts = batch["response"]
    toxicity_inputs = toxicity_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        ppo_trainer.accelerator.device
    )
    logits = toxicity_model(**toxicity_inputs).logits.float()
    toxicity_labels = (logits[:, 0]).tolist()

    rewards = [torch.tensor(output) for output in toxicity_labels]

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # Save model every 100 epochs
    if epoch % 100 == 0:
        if ppo_trainer.accelerator.is_main_process:
            ppo_trainer.save_pretrained(model_save_path)
            
ppo_trainer.save_pretrained(model_save_path)
