# Data

## Dowload Data

The MOSES dataset can be downloaded from https://github.com/molecularsets/moses.

## Preprocessing

### Preprocess the data

```
python preprocess.py --train ./train.txt --split 100 --jobs 120 --output ./moses-processed
```
This script will preprocess the training data (subgraph enumeration & tree decomposition), and save results into a list of files. We suggest you to use small value for `--split` if you are working with smaller datasets.

### Deriving Vocabulary
If you are running our code on a new dataset, you need to compute the vocabulary from your dataset.
To perform tree decomposition over a set of molecules, run
```
cd fast_jtnn
python mol_tree.py -i ./../../data/train.txt -v ./../../data/vocab.txt
```
This gives you the vocabulary of cluster labels over the dataset `train.txt`.

