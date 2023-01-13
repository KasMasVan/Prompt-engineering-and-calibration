Our code is based on code for the paper ["Surface Form Competition: Why the Highest Probability Answer Isn't Always Right"](https://peterwestuw.github.io/surface-form-competition-project/)

## Dependencies
We use python3 and pytorch 1.7.0, but we do not use cutting-edge features from either and expect to be largely forward and backward compatible. That is not a guarantee or promise.

You can use `pip install -r requirements.txt` to install the required libraries.

## Downloading Datasets
You may start by creating the data directory:
```
mkdir data
```
`DATA_README.md` has thorough instructions for downloading and processing datasets. We provide automatic downloaders and processers for datasets where possible in `data_downloaders/` but see `DATA_README` for full instructions.

## Downloading Models

You can go through `scripts/download_models.ipynb` to download models to `models/`.

## Reproducing Results
You can reproduce results from our paper by running following scripts:

```
bash ./scripts/tab2_gpt2.sh
bash ./scripts/tab7_gpt3.sh
bash ./scripts/tab8_t5.sh
bash ./scripts/tab9_flant5.sh
```
Each of them corresponds to one table in our paper, which includes results from one family of models.
Specifically, before running GPT-3 experiments, you should place your openai api at `api.key`.
If there is any confusion, simply look in `score_new.py` to see the details.
You can check the output from either the terminal or text files under `/results`.

## Hyperparameter Tuning
If you want to tune hyperparameters for Flan-T5 models, you can run the following scripts:
```
bash ./scripts/flant5_prefix.sh
bash ./scripts/flant5_a.sh
```
You can tweak arguments in these scripts to tune other models with different hyperparameters.

<!-- ## GPT3 Experiments
### Step0: Prerequisite
Install dependencies, and download datasets.

### Step1: API
place your openai api at "api.key"

### STEP2: Hyperparameter Tunning-Finding Reweighting Factor a
bash scripts/gpt3_find_a.sh

### STEP3: Hyperparameter Tunning-Finding Prefixes
bash scripts/gpt3_find_prefix.sh

### STEP4: Run on the Entire Dev Set.
bash scripts/gpt3_final.sh -->

