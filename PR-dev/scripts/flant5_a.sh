#!/bin/bash

for model in "google/flan-t5-small" "google/flan-t5-base" "google/flan-t5-large"
do
    for a in 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0
    do
        python score_new.py obqa --model $model --reweight $a
        python score_new.py copa --model $model --reweight $a
        python score_new.py cqa --model $model --reweight $a
    done
done
