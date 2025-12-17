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
    smoothing: Optional[float] = field(default=0.0)
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    num_pos: Optional[int] = field(default=6)
    num_neg: Optional[int] = field(default=6)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[float] = field(default=0.001),
    max_prompts_per_group: Optional[int] = field(default=500),
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj",
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

def main(args):
    
    idx_vec = []
    for i in range(args.per_device_train_batch_size):
        idx_vec.append(torch.cat((torch.ones(int(args.num_pos),1),torch.zeros(args.num_neg,1),2*torch.ones(int(args.num_pos),1),3*torch.ones(args.num_neg,1),4*torch.ones(int(args.num_pos),1)),dim=0))
        
    idx_vec = torch.cat(idx_vec)
    pos_idx = torch.where(idx_vec == 1)[0]
    neg_idx = torch.where(idx_vec == 0)[0]
    seq_pos_idx = torch.where(idx_vec == 2)[0]
    seq_neg_idx = torch.where(idx_vec == 3)[0]  
    cyc_idx = torch.where(idx_vec == 4)[0]  
    


    wandb_project = "Seq2SeqDetox"
    wandb_run_name = ""

    wandb.init(project=wandb_project)

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

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[item for item in args.lora_target_modules.split(',')],
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

    #args = Args()

    #args.data_dir = "/home/ubuntu/open-instruct/data/paraphrase/safeNLP_processed"
    #args.max_prompts_per_group = 500
    # args.use_llama_chat_format = False
    # args.use_chat_format = False

    #tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    pos_prompts = dict()
    neg_prompts = dict()

    groups = ['asian', 'black', 'chinese', 'jewish', 'latino', 'lgbtq', 'mental_dis', 'mexican', 'middle-eastern', 'muslim', 'native-american', 'physical_dis', 'women']
    groups = [f"neutral_{group}.json" for group in groups]
    dataset = load_dataset('json', data_dir=args.data_dir,data_files={'train': groups,}) 

    remove_id = []

    for idx,data in enumerate(tqdm(dataset['train'])):
        
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
            
        prompt = {'input_ids': prompt}
        
        prompt["label"] = data_point["id"]
        return prompt

    train_data = dataset.shuffle().map(generate_and_tokenize_prompt_pairs,fn_kwargs={"args":args}).remove_columns(["input", "id", "paraphrases", "paraphrases_toxic","group"])

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
            

                
                # circular
                pos_features = []
                neg_features = []
                post_features = []
                # add reference so it may be used
                _pos_list = self.pos_dict[feature["label"]] + [feature['input_ids']]
                pos_pool = np.arange(len(_pos_list))   
                for cls, nums, curr_dict in zip(["POS", "TRANS", "NEG"],[int(self.num_pos), int(self.num_pos),  self.num_hard_negs], [self.pos_dict, self.neg_dict, self.neg_dict]):
                    curr_pool = np.arange(len(curr_dict[feature["label"]]))
                    
                    np.random.shuffle(pos_pool)
                    
                    np.random.shuffle(curr_pool)
                    for i in range(nums):
                        item = curr_dict[feature["label"]][curr_pool[i]]
                        pos_item = _pos_list[i]
                        if cls == "POS": 
                            pos_features.append(tokenize(item))
                        if cls == "TRANS": # translate NEG to REF
                            prompt = f'<s>[INST] Paraphrase the following sentence, without using toxic or racist vocabulary: {item} [\INST] {pos_item}<\s>'
                            proxy_target = tokenize(prompt)

                            prompt = f'<s>[INST] Paraphrase the following sentence in a toxic way: {pos_item} [\INST] {item}<\s>'
                            proxy_target = tokenize(prompt)
                            post_features.append(proxy_target)
                        if cls == "NEG":
                            #translate from REF to NEG
                            prompt = f'<s>[INST] Paraphrase the following sentence in a toxic and possibly racist way: {pos_item} [\INST] {item}<\s>'
                            proxy_target = tokenize(prompt)
                            features.append(proxy_target)
                            neg_features.append(tokenize(item))
                features = features + pos_features + neg_features +  post_features

            features = self.tokenizer.pad(
                    features,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=return_tensors,
                )
            return features

    class CustomTrainer(transformers.Trainer):
        def __init__(self, num_pos, num_neg, batch_size, alpha, smoothing, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_pos = num_pos
            self.num_neg = num_neg
            self.batch_size = batch_size
            self.alpha = alpha
            self.smoothing = smoothing
        def compute_loss(self, model, inputs, return_outputs=False):
            

            # proxy_labels = torch.cat((inputs['labels'],tokenizer.eos_token_id * torch.ones((BS,np.abs(inputs['labels'].shape[1]-len_)),dtype=inputs['labels'].dtype).to(inputs['labels'].device)),dim=1)
            proxy_inputs = inputs['input_ids']
            proxy_masks = inputs['attention_mask']
            proxy_labels = inputs['input_ids']
            output = model(input_ids=proxy_inputs,attention_mask=proxy_masks,labels=proxy_labels, return_dict=True, loss_reduction="none")
            loss = einops.rearrange(output['loss'], " (B S) -> B S", B=inputs['input_ids'].shape[0])
            loss = torch.nan_to_num(loss, nan=5.0, posinf=5.0)
            if True:
                loss_fct = nn.CrossEntropyLoss()
                
                
                NUM_OFFSET =  args.num_pos +  args.num_neg
                NUM_POS = args.num_pos*3 # must be dividble by 2
                #NUM_SAMPLES = args.num_pos + 1 # + reference
                NUM_NEG =  args.num_neg*2# must be dividible by 2
                NUM_RAND_NEG = 0#int(args.num_pos/2)
                NUM_ELEMENTS = NUM_POS+NUM_NEG+NUM_RAND_NEG
                micro_batch_size = self.batch_size
                true_micro_batch_size = int(inputs['input_ids'].shape[0]/(NUM_POS+NUM_NEG+NUM_RAND_NEG))
                
                index_tensor = np.arange(true_micro_batch_size*NUM_ELEMENTS)#.reshape((true_micro_batch_size, NUM_ELEMENTS))
        
                firstLevel_pos_idx = index_tensor[seq_pos_idx].flatten()
                firstLevel_neg_idx = index_tensor[seq_neg_idx].flatten()
                
                ref_pos_avg = torch.mean(torch.mean(loss[firstLevel_pos_idx],dim=1))
                
                pos_scores_ = -torch.abs(torch.mean(loss[firstLevel_pos_idx],dim=1) - ref_pos_avg) / 1.0
                neg_scores_ = -torch.abs(torch.mean(loss[firstLevel_neg_idx],dim=1) - ref_pos_avg) / 1.0
                
                
                neg_scores_ = einops.rearrange(neg_scores_,"(a b)-> a b",a=micro_batch_size)
                pos_scores_ = einops.rearrange(pos_scores_,"(a b)-> a b",a=micro_batch_size)
                
                loss_fct = CrossEntropyLoss(label_smoothing=args.smoothing)
                final_scores = torch.cat([torch.pow(2.,pos_scores_), torch.pow(2.0, neg_scores_)], 1)
                label_probs_ = torch.cat([torch.ones_like(pos_scores_), torch.zeros_like(neg_scores_)],1).to(final_scores.device)
                constrastive_loss_perperplexity = loss_fct(final_scores, label_probs_) / (NUM_POS+1+NUM_RAND_NEG)
                
                loss_seq2seq_pos = torch.mean(loss,dim=1)[pos_idx]
                loss_seq2seq_neg = torch.mean(loss,dim=1)[neg_idx]
                loss_seq2seq_cyc = torch.mean(loss,dim=1)[cyc_idx]
                loss_seq2seq =  torch.mean(loss_seq2seq_cyc) + torch.mean(loss_seq2seq_neg) + torch.mean(loss_seq2seq_pos)# - torch.mean(loss_seq2seq_neg)
                loss_seqdiff = -torch.abs(torch.mean(loss_seq2seq_pos) - torch.mean(loss_seq2seq_neg))
                wandb.log({'train/seqdiff': loss_seqdiff.item(),'train/seq2seq': loss_seq2seq.item(), 'train/perplexity_loss': constrastive_loss_perperplexity.item()})
                
                return loss_seq2seq+self.alpha*constrastive_loss_perperplexity
            

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
                    tokenizer, pos_prompts, neg_prompts, num_pos=args.num_pos,num_hard_negs=args.num_neg,pad_to_multiple_of=8, return_tensors="pt", padding=True),
               
            num_pos=args.num_pos,
            num_neg=args.num_neg,
            batch_size=args.per_device_train_batch_size,
            alpha=args.alpha,
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