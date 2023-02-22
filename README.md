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
bash ./scripts/tiny_exp.sh
```
If there is any confusion, simply look in `score_new.py` to see the details.
You can check the output from either the terminal or text files under `/results`.