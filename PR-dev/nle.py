import os
import torch
import random
import json
import numpy as np
from tqdm import tqdm
from transformers import (GPT2LMHeadModel, 
                          GPT2Tokenizer)
from data_loaders import load_examples_cqa
import argparse

# reproducibility
def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def add_nle(test_examples, 
            nle_len=30, 
            temperature=0.7, 
            prefix="What is", 
            model=None, 
            device=None):
    """Add explanations for each answer before the qeustion."""
    
    assert model is not None
    assert device is not None
    
    model.to(device)
    
    for example in tqdm(test_examples):
        options = example['options']
        for option in options:
            # construct prompt for explanations
            answer = option['hypothesis']
            nle_prompt = prefix + answer + '?'

            # generate explanations
            input_ids = tokenizer(nle_prompt, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(
                input_ids = input_ids,
                max_length = len(input_ids) + nle_len,
                temperature= temperature,
                do_sample = True,
                num_return_sequences=1,
                pad_token_id = 50256,
            )

            # strip the generated text
            # generated_text = tokenizer.batch_decode(outputs)[0][len(nle_prompt):]
            generated_text = tokenizer.batch_decode(outputs)[0][:]
            generated_text = generated_text.strip()
            period_id = generated_text.find('.')
            if period_id != -1:
                generated_text = generated_text[:period_id + 1].strip()
            nle_text = generated_text
            option['premise'] = nle_text + option['premise']
    return test_examples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data/cqa/dev.jsonl')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='gpt2')
    parser.add_argument('--nle_len', type=int, default=30)
    parser.add_argument('--temperature', type=int, default=0.7)
    parser.add_argument('--prefix', type=str, default="What is")
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    print(args)
    if args.debug:
        import pdb; pdb.set_trace()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    
    # load what we need
    print("Loading examples: ")
    examples = load_examples_cqa(args.data_file)
    model = GPT2LMHeadModel.from_pretrained(args.ckpt)
    tokenizer = GPT2Tokenizer.from_pretrained(args.ckpt)
    
    # generate explanations
    test_examples = examples
    test_examples = add_nle(test_examples, 
                            nle_len = args.nle_len,
                            temperature=args.temperature,
                            prefix=args.prefix,
                            model=model, 
                            device=device,
                           )
    
    # save result to file
    save_path = f'data/cqa/dev_nle_w_{args.ckpt}_{args.seed}.jsonl'
    with open(save_path, 'w') as f:
        for example in test_examples:
            json.dump(example, f)
            f.write('\n')
    
    
    