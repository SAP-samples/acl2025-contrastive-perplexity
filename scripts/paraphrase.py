import argparse
import glob
import json
import os
import random
import re
from functools import partial
from multiprocessing import Pool
from typing import List, Optional

import numpy as np
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Generate paraphrases using LLM")
    parser.add_argument("--input_dir", type=str, default="data/eval/toxigen", help="Input directory containing text files")
    parser.add_argument("--output_dir", type=str, default="data/paraphrase/toxigen", help="Output directory for JSON files")
    parser.add_argument("--model_name", type=str, default="TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ", help="Model name or path")
    parser.add_argument("--model_basename", type=str, default="model", help="Model basename for GPTQ")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    parser.add_argument("--num_prompts", type=int, default=500, help="Number of prompts to process per group")
    parser.add_argument("--num_paraphrases", type=int, default=20, help="Number of paraphrases to generate per prompt")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    return parser.parse_args()

def jdump(obj, f_path, indent=4, default=str):
    """Dump a str or dictionary to a file in json format."""
    dir_name = os.path.dirname(f_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)
    with open(f_path, "w") as f:
        json.dump(obj, f, indent=indent, default=default)

def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.padding_side = "left"
    
    model = AutoGPTQForCausalLM.from_quantized(
        args.model_name,
        model_basename=args.model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device=args.device,
        use_triton=False,
        quantize_config=None
    )
    return model, tokenizer

def generate_paraphrases(args, model, tokenizer):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    
    # Process neutral files
    prompt_files = glob.glob(os.path.join(args.input_dir, "neutral*.txt"))
    
    for task_file in tqdm(prompt_files, desc="Processing files"):
        print(f"Processing {task_file}")
        
        with open(task_file, "r") as f:
            group_prompts = [line.strip() for line in f][:args.num_prompts]
            
        all_data = []
        group_name = os.path.basename(task_file).split(".")[0]
        
        for prompt in tqdm(group_prompts, desc="Prompts"):
            item = {"input": prompt}
            
            # Generate both neutral and toxic paraphrases
            for j in range(2):
                paraphrases = []
                sents = [x.strip() for x in [prompt] if x.strip()]
                
                for _ in trange(args.num_paraphrases, desc="Generating", leave=False):
                    sent_para = []
                    prompt_templates = []
                    
                    for sent in sents:
                        if j == 0:
                            postfix = ""
                            prompt_template = f'''### HUMAN: Paraphrase the following sentences: 
                            \"{sent}\"

                            ### RESPONSE:
                            '''
                        else:
                            postfix = "_toxic"
                            prompt_template = f'''### HUMAN: Paraphrase the following sentence in a very toxic way. make sure each sentence is toxic: 
                            \"{sent}\"

                            ### RESPONSE:
                            '''
                        prompt_templates.append(prompt_template)
                            
                    inputs = tokenizer(prompt_templates, return_tensors="pt", padding=True).to(args.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, 
                            penalty_alpha=0.3, 
                            top_k=25, 
                            max_new_tokens=150, 
                            do_sample=True
                        )
                    
                    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    
                    for k in range(len(response)):
                        # Simple parsing to extract response after ### RESPONSE:
                        if "### RESPONSE:" in response[k]:
                            filtered = response[k].split("### RESPONSE:")[-1].strip().lower()
                        else:
                            filtered = response[k].strip().lower()
                        sent_para.append(filtered)
                    
                    paraphrases.append(" ".join(sent_para))
                
                # Rank paraphrases by ROUGE score against input
                all_paraphrase_tokens = [scorer._tokenizer.tokenize(para) for para in paraphrases]
                ref_tokens = scorer._tokenizer.tokenize(" ".join(sents))
                
                # Calculate scores
                rouge_scores = []
                for para_tokens in all_paraphrase_tokens:
                    score = rouge_scorer._score_lcs(ref_tokens, para_tokens)
                    rouge_scores.append(score.fmeasure)
                
                # Select top similar paraphrases
                most_similar_paraphrases = {
                    paraphrases[i]: rouge_scores[i] 
                    for i in np.argsort(rouge_scores)[-15:][::-1]
                }
                
                item["paraphrases" + postfix] = most_similar_paraphrases
                
            all_data.append(item)
            
        # Save results
        out_file = os.path.join(args.output_dir, f"{group_name}.json")
        jdump(all_data, out_file)
        print(f"Saved results to {out_file}")

if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = load_model(args)
    generate_paraphrases(args, model, tokenizer)
