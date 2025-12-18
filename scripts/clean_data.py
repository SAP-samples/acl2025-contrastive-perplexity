#
# SPDX-FileCopyrightText: 2025 SAP SE or an SAP affiliate company
#
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import glob
import json
import os
import sys
from typing import List, Dict

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Clean data and compute toxicity scores")
    parser.add_argument("--src_dir", type=str, default="data/paraphrase/toxigen", help="Source directory containing JSON files")
    parser.add_argument("--output_dir", type=str, default="data/paraphrase/toxigen_processed", help="Output directory")
    parser.add_argument("--model_name", type=str, default="tomh/toxigen_roberta", help="Toxicity classifier model")
    parser.add_argument("--device", type=int, default=0, help="Device ID (e.g., 0 for cuda:0)")
    parser.add_argument("--groups", type=str, default="muslim,black,latino,asian,chinese,jewish,lgbtq,mental_dis,mexican,middle_east,native_american,physical_dis,women", help="Comma-separated list of groups to process")
    return parser.parse_args()

def jload(f_path):
    """Load a .json file into a dictionary."""
    with open(f_path, "r") as f:
        return json.load(f)

def jdump(obj, f_path, indent=4):
    """Dump a dictionary to a file in json format."""
    dir_name = os.path.dirname(f_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)
    with open(f_path, "w") as f:
        json.dump(obj, f, indent=indent, default=str)

def main(args):
    print(f"Loading toxicity classifier: {args.model_name}")
    device_str = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    
    cls_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    cls_model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    
    toxicity_classifier = pipeline(
        "text-classification", 
        model=cls_model, 
        tokenizer=cls_tokenizer, 
        device=args.device,
        return_all_scores=True
    )

    groups = [g.strip() for g in args.groups.split(",")]
    
    for group in groups:
        input_file = os.path.join(args.src_dir, f"neutral_{group}.json")
        if not os.path.exists(input_file):
            print(f"Warning: File {input_file} not found. Skipping.")
            continue
            
        print(f"Processing {group} from {input_file}")
        try:
            tmp_dict = jload(input_file)
        except Exception as e:
            print(f"Error loading {input_file}: {e}")
            continue

        processed_data = []
        for key, item in enumerate(tqdm(tmp_dict, desc=f"Scoring {group}")):
            try:
                item['group'] = group
                item['id'] = f"{group}-{key}"
                
                # Process both paraphrases and toxic paraphrases
                for misspelled, correct in zip(['paraphrases', 'paraphrases_toxic'], ['paraphrases', 'paraphrases_toxic']):
                    if misspelled in item:
                        # Extract texts
                        texts = list(item[misspelled].keys())
                        item[correct] = texts
                        
                        # Compute toxicity scores
                        scores = []
                        # Batch processing could be more efficient, but staying simple for now
                        for text in texts:
                            # Truncate to max length supported by model (usually 512)
                            result = toxicity_classifier(text, max_length=512, truncation=True)
                            # Assuming label 1 is toxic for this model
                            score = result[0][1]['score']
                            scores.append(score)
                            
                        item[f"{correct}_toxigenscore"] = scores
                        
                        # Remove old key if different
                        if misspelled != correct:
                            item.pop(misspelled)
                
                processed_data.append(item)
            except Exception as e:
                print(f"Error processing item inside group {group}: {e}")
                continue
                
        output_file = os.path.join(args.output_dir, f"neutral_{group}.json")
        jdump(processed_data, output_file)
        print(f"Saved processed data to {output_file}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
