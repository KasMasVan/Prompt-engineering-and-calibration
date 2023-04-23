import os
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
)

# gpt2_ckpts = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
# t5_ckpts = ['t5-small', 't5-base', 't5-large']
# flant5_ckpts = ['google/flan-t5-small', 'google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl']
gpt2_ckpts = ['gpt2-large', 'gpt2-xl']


for ckpt in gpt2_ckpts:
    tokenizer = GPT2Tokenizer.from_pretrained(ckpt)
    model = GPT2LMHeadModel.from_pretrained(ckpt)
    
    save_dir = f"../models/{ckpt}"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

# for ckpt in t5_ckpts:
    # tokenizer = AutoTokenizer.from_pretrained(ckpt)
    # model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    
    # save_dir = f"../models/{ckpt}"
    # os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    # model.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)

# for ckpt in flant5_ckpts:
#     tokenizer = AutoTokenizer.from_pretrained(ckpt)
#     model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    
#     save_dir = f"../models/{ckpt}"
#     os.makedirs(os.path.dirname(save_dir), exist_ok=True)
#     model.save_pretrained(save_dir)
#     tokenizer.save_pretrained(save_dir)