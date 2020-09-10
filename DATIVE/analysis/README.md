# Analysis of DATIVE dataset
## Calculate sentence probabilities

```{python3}
python CalcSentProbs.py [model_name]
```
is the basic command for calculating sentence probabilities.
`[model_name]` can be one of the following: `ngram`, `lstm`, `lstm-large`, `bert`, `gpt2`, and `gpt2-large`.

In addition to this, the following preparation is needed for the ngram and the LSTMs.

For `ngram`, you need `.arpa` file.  For details as to how to create `.arpa`, see [kenlm](https://github.com/kpu/kenlm).  We created `.arpa` file using the 80M Wikipedia data provided on [LM_syneval](https://github.com/TalLinzen/LM_syneval), which was also used for training the smaller LSTM we used for evaluation.

For `lstm`, clone [colorlessgreenRNNs](https://github.com/facebookresearch/colorlessgreenRNNs), and place `src/language_models/model.py` in `Models`.  In addition, download the checkpoint file, `hidden650_batch128_dropout0.2_lr20.0.pt` from [here](https://github.com/facebookresearch/colorlessgreenRNNs/tree/master/src), and put it in the `DATIVE/data` directory.

For `lstm-large`, create `lm_1b_data` directory under `DATIVE/data`.  Then download data files from [here](https://github.com/tensorflow/models/tree/master/research/lm_1b), and place them inside `lm_1b_data`.  In addition, clone the above repository and place `data_utils.py` in `Models`. 

In order to run the code for `lstm-large`, you need to use flags in the following way.
```{python3}
python CalcSentProbs.py lstm-large --pbtxt ../data/lm_1b_data/graph-2016-09-10.pbtxt --ckpt '../data/lm_1b_data/ckpt-*' --vocab_file ../data/lm_1b_data/vocab-2016-09-10.txt
```

The output will be added to `DATIVE/data/generated_pairs_with_results.csv`, which already contains the results we used for the paper.

## Extract hidden states
```{python3}
python ExtractHidden.py [model_name] [position_in_the_sentence]
```
is the basic command for extracting hidden states.
`[model_name]` can be one of the following: `lstm`, `lstm-large`, `gpt2`, and `gpt2-large`.
`[position_in_the_sentence]` can be one of the following: `verb`, `first_obj`, and `eos`.

For LSTMs, you need the same preparation as above.

In order to run the code for `lstm-large`, you need to use flags in the following way.
```{python3}
python ExtractHidden.py lstm-large --pbtxt lm_1b_data/graph-2016-09-10.pbtxt --ckpt 'lm_1b_data/ckpt-*' --vocab_file lm_1b_data/vocab-2016-09-10.txt
```

The output will be stored in `datafile` directory, and will be used for the ridge regression described below.

## Ridge regression
```{python3}
python RidgeRegression.py [model_name] [position_in_the_sentence] [id]
```
is the basic command for running the ridge regression.
`[model_name]` can be one of the following: `lstm`, `lstm-large`, `gpt2`, and `gpt2-large`.
`[position_in_the_sentence]` can be one of the following: `verb`, `first_obj`, and `eos`.
`[id]` is for running this code multiple times.  In our experiment, we ran this code 10 times (id = 0~9).

The output will be stored in `datafile` directory.

## Produce figures
Figure 1 is created by `behavioral.Rmd`, and Figures 2 and 3 are created by `model_eval.Rmd`.
`PlotReg.py` is a sample code for plotting the results for ridge regression.
