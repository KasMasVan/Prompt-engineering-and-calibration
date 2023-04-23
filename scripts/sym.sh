#!/bin/bash
# models=("google/flan-t5-small" "google/flan-t5-base" "google/flan-t5-large" "google/flan-t5-xl" 't5-small' 't5-base' 't5-large' "gpt2" "m" "l" "xl" )
models=("gpt2" "m" "l" "xl")
datasets_1=("cqa" "siqa")
# datasets_2=("copa" "obqa" "piqa")
cond_mcp_1="Given answers in square brackets [], choose the best for the question. Answers: [answers]. Question: "
# cond_mcp_2="Given answers in square brackets [], choose the one that best completes the sentence. Answers: [answers]. Sentence: "
uncond_mcp="Given answers in square brackets [], choose the best one. Answers: [answers]."
domain_cond_1=" The best answer is: "
# domain_cond_2=" The best completion is: "


for model in "${models[@]}"
do
    for dataset_1 in "${datasets_1[@]}"
    do
        # baseline
        # python score_new.py ${dataset_1} --model $model
        
        # mcp w/o symbol
        # python score_new.py ${dataset_1} \
        # --model ${model} \
        # --cond_mcp "$cond_mcp_1" \
        # --uncond_mcp "$uncond_mcp" \
        # --domain_cond "$domain_cond_1" 

        # mcp w/ symbol
        python score_new.py ${dataset_1} \
        --model ${model} \
        --cond_mcp "$cond_mcp_1" \
        --uncond_mcp "$uncond_mcp" \
        --domain_cond "$domain_cond_1" \
        --with_symbol 

    done
done
