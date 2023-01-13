python score_new.py obqa --model xl --prefix "Therefore, " --reweight 0.4;
python score_new.py cqa --model xl --prefix "Deductively, " --reweight 1.3;
python score_new.py copa --model xl --prefix "Consequently, " --reweight 2.0;