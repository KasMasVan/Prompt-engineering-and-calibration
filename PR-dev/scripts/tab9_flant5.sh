#!/bin/bash

# t5-small
# reweighting
python score_new.py obqa --model "google/flan-t5-small" --reweight 0.3
python score_new.py cqa --model "google/flan-t5-small" --reweight 3.0 
python score_new.py copa --model "google/flan-t5-small" --reweight 3.0
# reweighting + prefixing
python score_new.py obqa --model "google/flan-t5-small" --reweight 0.3 --prefix "Evidently, "
python score_new.py cqa --model "google/flan-t5-small" --reweight 3.0 --prefix "Obviously, "
python score_new.py copa --model "google/flan-t5-small" --reweight 3.0 --prefix "Abductively, "

# t5-base
# reweighting
python score_new.py obqa --model "google/flan-t5-base" --reweight 0.3
python score_new.py cqa --model "google/flan-t5-base" --reweight 2.9 
python score_new.py copa --model "google/flan-t5-base" --reweight 2.7
# reweighting + prefixing
python score_new.py obqa --model "google/flan-t5-base" --reweight 0.3 --prefix "Consequently, "
python score_new.py cqa --model "google/flan-t5-base" --reweight 2.9 --prefix "Therefore, "
python score_new.py copa --model "google/flan-t5-base" --reweight 2.7 --prefix "Consequently, "

# ada
# reweighting
python score_new.py obqa --model "google/flan-t5-large" --reweight 0.3
python score_new.py cqa --model "google/flan-t5-large" --reweight 2.3 
python score_new.py copa --model "google/flan-t5-large" --reweight 2.8
# reweighting + prefixing
python score_new.py obqa --model "google/flan-t5-large" --reweight 0.3 --prefix "Obviously, "
python score_new.py cqa --model "google/flan-t5-large" --reweight 2.3 --prefix "Finally, "
python score_new.py copa --model "google/flan-t5-large" --reweight 2.8 --prefix "Evidently, "