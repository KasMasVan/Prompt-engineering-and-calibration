#!/bin/bash

# for model_size in "gpt2" "m" "l" "xl"
# do
#     python score_new.py obqa --model $model_size --reweight 0.4 
#     python score_new.py obqa --model $model_size --prefix "Therefore, " --reweight 0.4
#     python score_new.py cqa --model $model_size --reweight 1.3
#     python score_new.py cqa --model $model_size --prefix "Deductively, " --reweight 1.3
#     python score_new.py copa --model $model_size --reweight 2.0
#     python score_new.py copa --model $model_size --prefix "Consequently, " --reweight 2.0
# done

for model_size in "gpt2" "m" "l" "xl"
do
    # baseline
    python score_new.py cqa --model $model_size
    # mcp
    python score_new.py cqa --model $model_size --mcp "Given [answers], "
done
