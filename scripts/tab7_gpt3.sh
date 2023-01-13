#!/bin/bash

# ada
# reweighting
python score_new.py obqa --model "ada" --reweight 0.7
python score_new.py cqa --model "ada" --reweight 1.1 
python score_new.py copa --model "ada" --reweight 1.5
# reweighting + prefixing
python score_new.py obqa --model "ada" --reweight 0.7 --prefix "Deductivley, "
python score_new.py cqa --model "ada" --reweight 1.1 --prefix "Inductively, "
python score_new.py copa --model "ada" --reweight 1.5 --prefix "Eventually, "