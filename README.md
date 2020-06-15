# neural_constructions
## Simulate the behavioral experiment
The `behavioral_experiment` directory contains necessary codes for running the behavioral experiment on a local computer.  The detailed instructions are described in the README in `behavioral_experiment`.

## Calculate sentence probabilities

```{python3}
python CalcSentProbs.py [model_name]
```
is the basic command for calculating sentence probabilities.
`[model_name]` can be one of the following: `ngram`, `lstm`, `lstm-large`, `bert`, `gpt2`, and `gpt2-large`.

In addition to this, the following preparation is needed for LSTMs.

For `lstm`, clone `https://github.com/facebookresearch/colorlessgreenRNNs`, and place `src/language_models/model.py` in the main repository, `neural_constructions`.  In addition, download the checkpoint file, `hidden650_batch128_dropout0.2_lr20.0.pt` from the page, `https://github.com/facebookresearch/colorlessgreenRNNs/tree/master/src`, and put it in the `datafile` repository.

For `lstm-large`, create `lm_1b_datafile` directory under `neural_constructions`.  Then download data files from `https://github.com/tensorflow/models/tree/master/research/lm_1b`, and place them inside `lm_1b_data`.  In addition, clone the above repository and include `data_utils.py` in `neural_constructions`. 

In order to run the code for `lstm-large`, you need to use flags in the following way.
`python CalcSentProbs.py --pbtxt lm_1b_data/graph-2016-09-10.pbtxt --ckpt 'lm_1b_data/ckpt-*' --vocab_file lm_1b_data/vocab-2016-09-10.txt`

## Extract hidden states
```{python3}
python ExtractHidden.py [model_name] [position_in_the_sentence]
```
is the basic command for extracting hidden states.
`[model_name]` can be one of the following: `lstm`, `lstm-large`, `gpt2`, and `gpt2-large`.
`[position_in_the_sentence]` can be one of the following: `verb`, `first_obj`, and `eos`.

For LSTMs, you need the same preparation as above.

In order to run the code for `lstm-large`, you need to use flags in the following way.
`python ExtractHidden.py --pbtxt lm_1b_data/graph-2016-09-10.pbtxt --ckpt 'lm_1b_data/ckpt-*' --vocab_file lm_1b_data/vocab-2016-09-10.txt`

## Ridge regression
```{python3}
python RidgeRegression.py [model_name] [position_in_the_sentence] [id]
```
is the basic command for running the ridge regression.
`[model_name]` can be one of the following: `lstm`, `lstm-large`, `gpt2`, and `gpt2-large`.
`[position_in_the_sentence]` can be one of the following: `verb`, `first_obj`, and `eos`.
`[id]` is for running this code multiple times.  In our experiment, we ran this code 10 times (id = 0~9).

## Produce figures
Figure 1 are created by `behavioral.Rmd`, and Figures 2 and 3 are by `model_eval.Rmd`.
`PlotReg.py` is a sample code for plotting the results for ridge regression.
