# Analysis of DATIVE dataset

## Setting up the models

The transformer models we consider require installing [Hugging Face](https://github.com/huggingface/transformers) API. 

The following preparation steps are needed for the ngram model and the LSTMs.

For `ngram`, you need the `.arpa` file.  For details as to how to create `.arpa`, see [kenlm](https://github.com/kpu/kenlm).  We created `.arpa` file using the 80M Wikipedia data provided on [LM_syneval](https://github.com/TalLinzen/LM_syneval), which was also used for training the smaller LSTM we used for evaluation.

For `lstm`, clone [colorlessgreenRNNs](https://github.com/facebookresearch/colorlessgreenRNNs), and place `src/language_models/model.py` in this directory.  In addition, download the checkpoint file, `hidden650_batch128_dropout0.2_lr20.0.pt` from [here](https://github.com/facebookresearch/colorlessgreenRNNs/tree/master/src), and put it in the `models` directory.

For `lstm-large`, create an empty `lm_1b_data` directory under `Models`.  Then follow the installation steps described in the [lm_1b](https://github.com/tensorflow/models/tree/archive/research/lm_1b) README, placing all data files inside `lm_1b_data`.  

## Calculate sentence probabilities

```{python3}
python CalcSentProbs.py [model_name]
```
is the main script for calculating sentence probabilities.

`[model_name]` can be one of the following models: `ngram`, `lstm`, `lstm-large`, `bert`, `gpt2`, and `gpt2-large`.

In order to run the code for `lstm-large`, you need to use flags in the following way.

```{python3}
python CalcSentProbs.py lstm-large --pbtxt ../../models/lm_1b_data/graph-2016-09-10.pbtxt --ckpt '../../models/lm_1b_data/ckpt-*' --vocab_file ../../models/lm_1b_data/vocab-2016-09-10.txt
```

The output will be added as a new column of `DATIVE/data/generated_pairs_with_results.csv`, which contains the results we used for the paper.

## Extract hidden states
```{python3}
python ExtractHidden.py [model_name] [position_in_the_sentence]
```
is the basic command for extracting hidden states.
`[model_name]` can be one of the following: `lstm`, `lstm-large`, `gpt2`, and `gpt2-large`.
`[position_in_the_sentence]` can be one of the following: `verb`, `first_obj`, and `eos`.

For LSTMs, you need the same preparation as above.

Again, in order to run the code for `lstm-large`, you need to use flags in the following way.
```{python3}
python ExtractHidden.py lstm-large --pbtxt ../../models/lm_1b_data/graph-2016-09-10.pbtxt --ckpt '../../models/lm_1b_data/ckpt-*' --vocab_file ../../models/lm_1b_data/vocab-2016-09-10.txt
```

The output will be stored in `datafile` directory, and is used for the ridge regression described below.

## Ridge regression
```{python3}
python RidgeRegression.py [model_name] [position_in_the_sentence] [id]
```
is the basic command for running the ridge regression.
`[model_name]` can be one of the following: `lstm`, `lstm-large`, `gpt2`, and `gpt2-large`.
`[position_in_the_sentence]` can be one of the following: `verb`, `first_obj`, and `eos`.
`[id]` is for running this code multiple times.  In our experiment, we ran this code 10 times (id = 0~9).

The output will be stored in `datafile` directory.

## Reproduce figures and statistical results

Figure 1 is created by `behavioral.Rmd`, and Figures 2 and 3 are created by `model_eval.Rmd`.
`PlotReg.py` is sample code for plotting the results for ridge regression.
