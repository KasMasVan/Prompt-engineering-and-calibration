#!/bin/bash

# python score_new.py obqa \
#     --model gpt2 \
#     --cond_mcp "Given answers in [], choose the one that best completes the sentence. Answers: [answers], Sentence: " \
#     --uncond_mcp "Given answers in [], choose the best one. Answers: [answers]." \
#     --domain_cond " The best completion is:" \
#     --debug 
    
    

for model in "gpt2" "m" "l" "xl" "google/flan-t5-small" "google/flan-t5-base" "google/flan-t5-large" "google/flan-t5-xl"
do
    # baseline
    # python score_new.py copa --model $model
    
    # mcp
    python score_new.py obqa \
    --model $model \
    --cond_mcp "Given answers in [], choose the one that best completes the sentence. Answers: [answers], Sentence: " \
    --uncond_mcp "Given answers in [], choose the best one. Answers: [answers]." \
    --domain_cond " The best completion is:" \
done
