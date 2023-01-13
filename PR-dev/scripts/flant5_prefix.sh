 #!/bin/bash

# for model in "google/flan-t5-small" "google/flan-t5-base" "google/flan-t5-large"

for prefix in "Inductively, " "Deductively, " "Abductively, " "Therefore, " "Apparently, " "Obviously, " "Consequently, " "Evidently, " "Finally, " "Eventually, "
do
    # small: 0.3;3.0;3.0 
    python score_new.py obqa --model "google/flan-t5-small" --reweight 0.3 --prefix $prefix
    python score_new.py cqa --model "google/flan-t5-small" --reweight 3.0 --prefix $prefix
    python score_new.py copa --model "google/flan-t5-small" --reweight 3.0 --prefix $prefix
done

for prefix in "Inductively, " "Deductively, " "Abductively, " "Therefore, " "Apparently, " "Obviously, " "Consequently, " "Evidently, " "Finally, " "Eventually, "
do
    # base: 0.3;2.9;2.7
    python score_new.py obqa --model "google/flan-t5-base" --reweight 0.3 --prefix $prefix
    python score_new.py cqa --model "google/flan-t5-base" --reweight 2.9 --prefix $prefix
    python score_new.py copa --model "google/flan-t5-base" --reweight 2.7 --prefix $prefix
done

for prefix in "Inductively, " "Deductively, " "Abductively, " "Therefore, " "Apparently, " "Obviously, " "Consequently, " "Evidently, " "Finally, " "Eventually, "
do
    # large: 0.3;2.3;2.8
    python score_new.py obqa --model "google/flan-t5-large" --reweight 0.3 --prefix $prefix
    python score_new.py cqa --model "google/flan-t5-large" --reweight 2.3 --prefix $prefix
    python score_new.py copa --model "google/flan-t5-large" --reweight 2.8 --prefix $prefix
done