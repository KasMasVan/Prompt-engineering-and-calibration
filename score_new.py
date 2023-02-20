def get_model(model_name, key_file):
    if model_name.lower() in ['gpt2', 'gpt2-s', 'gpt2-small', 'gs', 's', 'small']:
        # GPT-2 Small
        model   = GPT2LMHeadModel.from_pretrained('./models/gpt2').cuda(0).eval()
        encoder = GPT2Tokenizer.from_pretrained('./models/gpt2')
        name    = 'G-S'
    elif model_name.lower() in ['gpt2-m', 'gpt2-medium', 'gm', 'm', 'medium']:
        # GPT-2 Medium
        model   = GPT2LMHeadModel.from_pretrained('./models/gpt2-medium').cuda(0).eval()
        encoder = GPT2Tokenizer.from_pretrained('./models/gpt2-medium')
        name    = 'G-M'
    elif model_name.lower() in ['gpt2-l', 'gpt2-large', 'gl', 'l', 'large']:
        # GPT-2 Large
        model   = GPT2LMHeadModel.from_pretrained('./models/gpt2-large').cuda(0).eval()
        encoder = GPT2Tokenizer.from_pretrained('./models/gpt2-large')
        name    = 'G-L'
    elif model_name.lower() in ['gpt2-xl', 'gxl', 'xl', 'extra-large']:
        # GPT-2 XL
        model   = GPT2LMHeadModel.from_pretrained('./models/gpt2-xl').cuda(0).eval()
        encoder = GPT2Tokenizer.from_pretrained('./models/gpt2-xl')
        name    = 'G-XL'
    elif "gpt-neo" in model_name:
        # "EleutherAI/gpt-neo-125M"
        # "EleutherAI/gpt-neo-1.3B"
        # "EleutherAI/gpt-neo-2.7B"
        model = AutoModelForCausalLM.from_pretrained(f"./models/{model_name}")
        encoder = AutoTokenizer.from_pretrained(f"./models/{model_name}")
        name = "GPT-Neo"
    elif model_name in ['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b']:
        model = AutoModelForSeq2SeqLM.from_pretrained(f"./models/{model_name}")
        encoder = AutoTokenizer.from_pretrained(f"./models/{model_name}")
        name = "T5"
    elif model_name in ['google/flan-t5-small', 'google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl', 'google/flan-t5-xxl']:
        model = AutoModelForSeq2SeqLM.from_pretrained(f"./models/{model_name}")
        encoder = AutoTokenizer.from_pretrained(f"./models/{model_name}")
        name = "Flan-T5"
    elif model_name.lower() == 'ada' or \
         model_name.lower() == 'babbage' or \
         model_name.lower() == 'curie' or \
         model_name.lower() == 'davinci':
        # GPT-3
        model = name = model_name
        encoder = None
        import openai
        with open(key_file) as f:
            api_key = f.read().strip()
        openai.api_key = api_key
    else:
        raise ValueError(f'No model {model_name}')
    return model, encoder, name

def get_examples(dataset_name, split, stem, n_shot, variant, data_file, prefix, **kwargs):
    if dataset_name == 'copa':
        from data_loaders_new import load_examples_copa_prefix, load_examples_copa_mcp
        if kwargs["cond_mcp"] != "" or kwargs["uncond_mcp"] != "" or kwargs["domain_cond"] !="":
            examples = load_examples_copa_mcp(f'{stem}copa-{split}.xml', **kwargs)
        else:
            examples = load_examples_copa_prefix(f'{stem}copa-{split}.xml', prefix=prefix)
        closed_label_space = False
        
    elif dataset_name == 'obqa':
        from data_loaders_new import load_examples_obqa_prefix, load_examples_obqa_mcp
        if kwargs["cond_mcp"] != "" or kwargs["uncond_mcp"] != "" or kwargs["domain_cond"] !="":
            examples = load_examples_obqa_mcp(f'{stem}{split}.jsonl', **kwargs)
        else:
            examples = load_examples_obqa_prefix(f'{stem}{split}.jsonl', prefix=prefix)
        closed_label_space = False
        
    elif dataset_name == 'cqa':
        from data_loaders_new import load_examples_cqa_with_nle, load_examples_cqa, load_examples_cqa_prefix, load_examples_cqa_mcp
        if kwargs["cond_mcp"] != "" or kwargs["uncond_mcp"] != "" or kwargs["domain_cond"] !="":
            # apply multiple choice prompt.
            examples = load_examples_cqa_mcp(f'{stem}{split}.jsonl', **kwargs)
        else:
            # apply prefix, i.e., normal prompt.
            examples = load_examples_cqa_prefix(f'{stem}{split}.jsonl', prefix=prefix)

        closed_label_space = False
    elif dataset_name == "storycloze":
        from data_loaders_new import load_examples_storycloze_prefix
        examples = load_examples_storycloze_prefix(f'{stem}{split}.tsv', prefix=prefix)
        closed_label_space = False
    elif dataset_name == 'boolq':
        from data_loaders_new import load_examples_boolq
        examples = load_examples_boolq(f'{stem}dev.jsonl', **kwargs)
        closed_label_space = True
    
    elif dataset_name == 'piqa':
        from data_loaders_new import load_examples_piqa_mcp
        examples = load_examples_piqa_mcp(f'{stem}valid.jsonl', f'{stem}valid-labels.lst', **kwargs)
        closed_label_space = False
    elif dataset_name == 'siqa':
        from data_loaders_new import load_examples_siqa_mcp
        examples = load_examples_siqa_mcp(f'{stem}dev.jsonl', f'{stem}dev-labels.lst', **kwargs)
        closed_label_space = False
    
    return examples, closed_label_space


if __name__ == '__main__':
    from transformers import (GPT2LMHeadModel, 
                              GPT2Tokenizer,
                              AutoTokenizer,
                              AutoModelForCausalLM,
                              AutoModelForSeq2SeqLM,
                             )
    from utils_new import score
    import argparse
    import random
    import numpy as np
    import torch
    import os
    import pdb

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--model', type=str, default='xl')
    parser.add_argument('--n-shot', type=int, default=0)
    parser.add_argument('--variant', type=int, default=None)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--key', type=str, default='api.key')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--domain_cond', type=str, default="") # use this with care
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument("--reweight", type=float, default=1.0)
    parser.add_argument("--do_max_sat", action="store_true", help="Whether to invoke a MaxSAT solver to re-rank predictions.")
    parser.add_argument("--multiple_target_words", action="store_true", help="Enable multiple target words. For closed set tasks, e.g., BoolQ.")
    parser.add_argument("--small", action="store_true", help="Whether to load the small version for tuning hyperparameters.")
    parser.add_argument("--cond_mcp", type=str, default="", help="Multiple choice prompt for conditional premesis.")
    parser.add_argument("--uncond_mcp", type=str, default="", help="Multiple choice prompt for unconditional premesis.")
    args = parser.parse_args()
    print(args)

    if args.debug:
        pdb.set_trace()

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, encoder, name = get_model(args.model, args.key)
    if args.dataset.endswith('-rev'):
        stem = f'data/{args.dataset[:-4]}/'
    else:
        stem = f'data/{args.dataset}/'
    examples, closed_label_space = get_examples(args.dataset, args.split, stem, args.n_shot, args.variant, args.data_file, args.prefix, multiple_target_words=args.multiple_target_words, small=args.small, cond_mcp=args.cond_mcp, uncond_mcp=args.uncond_mcp, domain_cond=args.domain_cond)
    if args.sample:
        assert(args.sample <= len(examples))
        examples = random.sample(examples, args.sample)
    accs = score(model, args.model, encoder, examples, stem, args.split, args.batch, args.reweight, args.do_max_sat, multiple_target_words=args.multiple_target_words)
    
    for key, value in accs.items():
        accs[key] = f"{value:.3f}"
    
    # print results
    # with open(f"data/{args.dataset}/many_runs.txt", 'a') as f:
    save_path = f"results/{args.dataset}/many_runs_{args.dataset}_{args.model.split('/')[-1]}.txt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'a') as f:
        print(args, file=f)
        if len(args.prefix) == 0:
            # reweighting only
            for key, value in accs.items():
                if "DC_a" not in key:
                    print(key, value)
                    print(key, value, file=f)
        else:
            # reweighitng + prefixing
            key_1 = 'PMI_a'
            key_2 = 'PMI_DC_a'
            key_3 = 'PMI_DC'
            print('PMI_PR', accs[key_1])
            print('PMI_PR', accs[key_1], file=f)
            print('PMI_DC_prefix', accs[key_3])
            print('PMI_DC_prefix', accs[key_3], file=f)
            print("PMI_DC_PR", accs[key_2])
            print("PMI_DC_PR", accs[key_2], file=f)