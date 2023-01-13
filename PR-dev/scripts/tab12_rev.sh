for data in "obqa" "cqa" "copa"
do
    python score_new.py $data --model gpt2
done
