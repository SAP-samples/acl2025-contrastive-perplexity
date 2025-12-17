import argparse
import itertools
import json
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from models.patch_utils import apply_patch_if_needed

def parse_args():
    parser = argparse.ArgumentParser(description="Compute perplexity of model on data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or huggingface model name")
    parser.add_argument("--data_dir", type=str, default="data/paraphrase/safeNLP_processed", help="Path to data directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num_toxic_samples", type=int, default=4, help="Number of toxic samples to include")
    parser.add_argument("--num_neutral_samples", type=int, default=0, help="Number of neutral samples to include")
    parser.add_argument("--stride", type=int, default=512, help="Stride for perplexity calculation")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main(args):
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Loading model from {args.model_path}...")
    
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        quantization_config=quantization_config,
        device_map=args.device if not args.use_4bit else "auto",
        trust_remote_code=True
    )
    
    # Apply patch if needed (though for eval standard model forward is usually fine, 
    # unless we specifically need per-token losses which this script does manually anyway)
    # But let's apply for consistency if needed later
    # model = apply_patch_if_needed(model)
    
    print("Loading dataset...")
    groups = ['asian', 'black', 'chinese', 'jewish', 'latino', 'lgbtq', 
              'mental_dis', 'mexican', 'middle-eastern', 'muslim', 
              'native-american', 'physical_dis', 'women']
    data_files = [f"neutral_{group}.json" for group in groups]
    # Check if files exist
    valid_files = [f for f in data_files if os.path.exists(os.path.join(args.data_dir, f))]
    
    if not valid_files:
        print(f"Error: No data files found in {args.data_dir}")
        return

    dataset = load_dataset('json', data_dir=args.data_dir, data_files={'train': valid_files})
    
    list1d = []
    
    if args.num_toxic_samples > 0:
        print(f"Sampling Mixture - Toxic: {args.num_toxic_samples}, Neutral: {args.num_neutral_samples}")
        list2d = []
        for x in dataset['train']:
            toxic_samples = []
            neutral_samples = []
            
            if 'paraphrases_toxic' in x and len(x['paraphrases_toxic']) > 0:
                toxic_samples = random.sample(x['paraphrases_toxic'], min(args.num_toxic_samples, len(x['paraphrases_toxic'])))
            
            if 'paraphrases' in x and len(x['paraphrases']) > 0 and args.num_neutral_samples > 0:
                neutral_samples = random.sample(x['paraphrases'], min(args.num_neutral_samples, len(x['paraphrases'])))
                
            list2d.append(toxic_samples + neutral_samples)
            
        list1d = list(itertools.chain(*list2d))
    else:
        print("Sampling Positive (Neutral) only")
        # Default behavior when num_toxic is 0
        num_samples = max(4, args.num_neutral_samples)
        list2d = []
        for x in dataset['train']:
            if 'paraphrases' in x and len(x['paraphrases']) > 0:
                list2d.append(random.sample(x['paraphrases'], min(num_samples, len(x['paraphrases']))))
        list1d = list(itertools.chain(*list2d))

    if not list1d:
        print("No samples found to evaluate.")
        return

    print(f"Evaluating perplexity on {len(list1d)} samples...")
    encodings = tokenizer("\n\n".join(list1d), return_tensors="pt")
    
    max_length = 4096 # model.config.n_positions if hasattr(model.config, 'n_positions') else 4096
    stride = args.stride
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.use_4bit:
        # Model is already on device handled by accelerate/bitsandbytes
        pass
    else:
        model = model.to(device)

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if len(nlls) > 0:
        ppl = torch.exp(torch.stack(nlls).mean())
        print(f"Model: {args.model_path}")
        print(f"Perplexity: {ppl.item():.4f}")
    else:
        print("Could not compute perplexity.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
