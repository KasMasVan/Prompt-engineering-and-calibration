#!/bin/bash

for model_size in "gpt2" "m" "l" "xl"
do
    # baseline
    # python score_new.py cqa --model $model_size
    # mcp
    python score_new.py cqa --model $model_size --mcp "Given [answers],"
done
