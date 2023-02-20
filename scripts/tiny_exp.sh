#!/bin/bash

python score_new.py siqa \
    --model gpt2 \
    --cond_mcp "Given answers in square brackets [], choose the best for the question. Answer: [answers]. Question: " \
    --uncond_mcp "Given answers in square brackets [], choose the best one. Answer: [answers]." \
    --domain_cond " The best answer is:" \
    --debug
    
    

# for model in 't5-small' 't5-base' 't5-large' #"gpt2" "m" "l" "xl" "google/flan-t5-small" "google/flan-t5-base" "google/flan-t5-large" "google/flan-t5-xl"
# do
#     # baseline
#     python score_new.py cqa --model $model
#     python score_new.py copa --model $model
#     python score_new.py obqa --model $model
#     python score_new.py piqa --model $model
#     python score_new.py siqa --model $model
    
#     # mcp
#     python score_new.py cqa \
#     --model $model \
#     --cond_mcp "Given answers in square brackets [], choose the best for the question. Answer: [answers]. Question: " \
#     --uncond_mcp "Given answers in square brackets [], choose the best one. Answer: [answers]." \
#     --domain_cond " The best answer is:" 
    
#     python score_new.py copa \
#     --model $model \
#     --cond_mcp "Given answers in [], choose the one that best completes the sentence. Answers: [answers], Sentence: " \
#     --uncond_mcp "Given answers in [], choose the best one. Answers: [answers]." \
#     --domain_cond " The best completion is:" 
    
#     python score_new.py obqa \
#     --model $model \
#     --cond_mcp "Given answers in [], choose the one that best completes the sentence. Answers: [answers], Sentence: " \
#     --uncond_mcp "Given answers in [], choose the best one. Answers: [answers]." \
#     --domain_cond " The best completion is:" 

#     python score_new.py piqa \
#     --model $model \
#     --cond_mcp "Given answers in [], choose the one that best completes the sentence. Answers: [answers], Sentence: " \
#     --uncond_mcp "Given answers in [], choose the best one. Answers: [answers]." \
#     --domain_cond " The best completion is:" 

#     # mcp
#     python score_new.py siqa \
#     --model $model \
#     --cond_mcp "Given answers in square brackets [], choose the best for the question. Answer: [answers]. Question: " \
#     --uncond_mcp "Given answers in square brackets [], choose the best one. Answer: [answers]." \
#     --domain_cond " The best answer is:" 

# done