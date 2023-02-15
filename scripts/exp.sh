#!/bin/bash

# python score_new.py cqa \
#     --model gpt2 \
#     --cond_mcp "Given [answers]," \
#     --uncond_mcp "Given [answers]," \
#     --domain_cond " the best answer is:" \
#     --debug \
    

for model_size in "gpt2" "m" "l" "xl"
do
    # baseline
    # python score_new.py cqa --model $model_size
    # mcp
    python score_new.py cqa \
    --model $model_size \
    --cond_mcp "Given [answers]," \
    --uncond_mcp "Given [answers]," \
    --domain_cond " the best answer is:" \
done
