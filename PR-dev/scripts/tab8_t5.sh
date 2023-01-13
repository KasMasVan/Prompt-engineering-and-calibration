#!/bin/bash

# t5-small
# reweighting
python score_new.py obqa --model "t5-small" --reweight 0.4
python score_new.py cqa --model "t5-small" --reweight 2.1 
python score_new.py copa --model "t5-small" --reweight 2.1
# reweighting + prefixing
python score_new.py obqa --model "t5-small" --reweight 0.4 --prefix "Deductively, "
python score_new.py cqa --model "t5-small" --reweight 2.1 --prefix "Abductively, "
python score_new.py copa --model "t5-small" --reweight 2.1 --prefix "Obviously, "

# t5-base
# reweighting
python score_new.py obqa --model "t5-base" --reweight 0.4
python score_new.py cqa --model "t5-base" --reweight 2.1 
python score_new.py copa --model "t5-base" --reweight 2.1
# reweighting + prefixing
python score_new.py obqa --model "t5-base" --reweight 0.4 --prefix "Consequently, "
python score_new.py cqa --model "t5-base" --reweight 2.1 --prefix "Evidently, "
python score_new.py copa --model "t5-base" --reweight 2.1 --prefix "Obviously, "

# ada
# reweighting
python score_new.py obqa --model "t5-large" --reweight 0.4
python score_new.py cqa --model "t5-large" --reweight 2.1 
python score_new.py copa --model "t5-large" --reweight 2.1
# reweighting + prefixing
python score_new.py obqa --model "t5-large" --reweight 0.4 --prefix "Finally, "
python score_new.py cqa --model "t5-large" --reweight 2.1 --prefix "Evidently, "
python score_new.py copa --model "t5-large" --reweight 2.1 --prefix "Obviously, "