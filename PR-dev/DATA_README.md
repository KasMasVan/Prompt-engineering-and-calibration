# Instructions for Downloading Data

Here we include instructions for downloading each dataset, for maximum reproducibility. Note that we do not download/process test sets where numbers are reported on the dev set, in order to follow previous work; please see the paper for details.

## Directory Structure

The location of datasets is hard-coded in `score_new.py`; this is a feature, not a bug, because it is easy to lose track of which data files have what in them. All datasets are stored in `data/<dataset_name>/`. To apply to a new dataset, add a data loader to `data_loaders.py` and add a couple lines of logic code to `score_new.py`.

## Datasets

### COPA

The official website for COPA is: https://people.ict.usc.edu/~gordon/copa.html and you can download copa using:

```
mkdir data/copa/
cd data/copa/
wget https://people.ict.usc.edu/~gordon/downloads/COPA-resources.tgz
tar -xvf COPA-resources.tgz
cp COPA-resources/datasets/copa-{dev,test}.xml .
```

or simply use `bash data_downloaders/copa.sh`


### OpenBookQA

The official website of OpenBookQA is: https://allenai.org/data/open-book-qa You can download OpenBookQA using:

```
mkdir data/obqa
cd data/obqa
wget https://ai2-public-datasets.s3.amazonaws.com/open-book-qa/OpenBookQA-V1-Sep2018.zip
unzip OpenBookQA-V1-Sep2018.zip
cp OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl dev.jsonl
cp OpenBookQA-V1-Sep2018/Data/Main/test.jsonl test.jsonl
```

or simply use `bash data_downloaders/obqa.sh`

### CommonsenseQA

The official website of CommonsenseQA is https://www.tau-nlp.org/commonsenseqa You can download the validation file using:

```
mkdir data/cqa/
cd data/cqa/
wget https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl -O dev.jsonl
```

or simply use `bash data_downloaders/cqa.sh`