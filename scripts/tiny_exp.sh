#!/bin/bash    

# for model in "google/flan-t5-small" "google/flan-t5-base" "google/flan-t5-large" "google/flan-t5-xl" 't5-small' 't5-base' 't5-large' "gpt2" "m" "l" "xl" 
for model in "google/flan-t5-small" "google/flan-t5-base" 't5-small' 't5-base' "gpt2" "m"
do
    # baseline
    python score_new.py cqa --model $model
    python score_new.py copa --model $model
    python score_new.py obqa --model $model
    python score_new.py piqa --model $model
    python score_new.py siqa --model $model
    
    # # mcp
    python score_new.py cqa \
    --model $model \
    --cond_mcp "Given answers in square brackets [], choose the best for the question. Answers: [answers]. Question: " \
    --uncond_mcp "Given answers in square brackets [], choose the best one. Answers: [answers]." \
    --domain_cond " The best answer is: " 
    
    python score_new.py siqa \
    --model $model \
    --cond_mcp "Given answers in square brackets [], choose the best for the question. Answers: [answers]. Question: " \
    --uncond_mcp "Given answers in square brackets [], choose the best one. Answers: [answers]." \
    --domain_cond " The best answer is: " 
    
    python score_new.py copa \
    --model $model \
    --cond_mcp "Given answers in square brackets [], choose the one that best completes the sentence. Answers: [answers]. Sentence: " \
    --uncond_mcp "Given answers in square brackets [], choose the best one. Answers: [answers]." \
    --domain_cond " The best completion is: " 
    
    python score_new.py obqa \
    --model $model \
    --cond_mcp "Given answers in square brackets [], choose the one that best completes the sentence. Answers: [answers]. Sentence: " \
    --uncond_mcp "Given answers in square brackets [], choose the best one. Answers: [answers]." \
    --domain_cond " The best completion is: " 
    
    python score_new.py piqa \
    --model $model \
    --cond_mcp "Given answers in square brackets [], choose the one that best completes the sentence. Answers: [answers]. Sentence: " \
    --uncond_mcp "Given answers in square brackets [], choose the best one. Answers: [answers]." \
    --domain_cond " The best completion is: " 

done
