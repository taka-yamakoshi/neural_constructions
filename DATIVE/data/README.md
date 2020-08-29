# DATIVE dataset
This directory contains the final form of DATIVE dataset as well as information as to how we constucted the dataset.
`verblists.csv` contains verb-theme pairs we used.  For each verb, there are five themes: two definite nouns, two indefinite nouns and the word "something".  The two definite nouns and the two indefinite nouns are hand-picked for semantic coherence.

`preprocess.Rmd` is used for creating `generated_pairs.csv` using the verb-theme pairs in `verblists.csv`.

`generated_pairs_with_results.csv` include the result of the behavioral experiment (under `BehavDOpreference`) and log-likelihood ratios for each model.  
