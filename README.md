
This repository contains the source code used by Tübingen-Oslo team in
[SemEval-2018](http://alt.qcri.org/semeval2018/)
[Multilingual Emoji Prediction Shared Task](https://competitions.codalab.org/competitions/17344).
The approach won the best place on both English and Spanish data sets.

The study is described in the paper:

- Çağrı Çöltekin and Taraka Rama (2018) [Tübingen-Oslo at SemEval-2018
    Task 2: SVMs perform better than RNNs at Emoji Prediction](semeval2018paper.pdf).
    In: Proceedings of the 12th International Workshop on Semantic Evaluation
    (SemEval-2018),  pages 34-38 ([bibtex](semeval2018paper.bib))

## Requirements

The repository includes only the code for the models.
You need to get the data following the instructions at
the [shared task](https://competitions.codalab.org/competitions/17344) web page.
For testing, a small sample is included in the repository.

Except standard Python libraries,
[scikit-learn](scikit-learn.org/),
[Keras](https://keras.io/)
(and [Tensorflow](https://www.tensorflow.org/))
is required to run scripts in this repository.

## A brief explanation of the scripts

All scripts run with Python 3 (may require some changes to run on
Python 2). Most of the scripts are controlled thorough command line
options and support `-h` option that gives a summary.

- `k-fold-linear.py` runs k-fold cross validation using a
    bag-of-n-grams model with given hyperparameters on specified input,
    and reports .
    `k-fold-linear.py -h` gives a brief usage information.

    For example,
    `python3 ./k-fold.py -r 0.1 -L word -f 2 -C 6 -W 2 -i data/sample`
    trains/tests an SVM (default) classifier using a combination of character
    n-grams of 1 to 6 (`-C`), word n-grams of 1 to 2 (`-W`);
    it excludes n-grams with document frequency one (`-f`).
    The features are weighted with TF-IDF (default).
    The SVM margin parameter `C` is set to 0.1 (`-r`).
    The input is specified with its prefix, since the shared task data
    has sparate files for the text and the labels.
    For the above command,
    files `data/sample.labels` and `data/sample.text` should
    exist in the format specified by the shared task data description.

    To run a grid (or random) search for finding a good hyperparameter setting,
    one run this script, looping around a set of hyperparameter values.
    To avoid re-calculating/re-weighting the features,
    this script, by default, saves the "vectorizer" in directory `.cache`
    and will use a cached vectorizer if a matching one exists in the cache.
    The cached data is never cleaned,
    and may take considerable space if a large hyperparameter space is explored.
    A sample UNIX shell script for doing a grid search is included as
    `grid-search-svm.sh`.

- `predict-linear.py` takes a set of command line parameters
    similar to `k-fold-linear.py` and a test file.
    It trains the model with the specified hyperparameters and the training file,
    and outputs the predicted labels.
    Output is simply a label-per-line with the same order
    as the corresponding texts in the test file.

- `rnn.py` contains the code for the RNN-based classifier reported in
    the paper.

- `tune-rnn.py` is wrapper around the `rnn.py` that does a random search
  over indicated hyperparameter ranges (specified at the top of the file),
  and outputs the validation score for each setting.

- `predict-rnn.py`, similar to `predict-linear.py`,
    trains an RNN model and outputs the predicted labels on the
    indicated test set.

- The scripts `svm-incremental.py` and `rnn-incremental.py` run the
    corresponding models with the given parameter setting,
    with increasing amount of training data and outputs the validation set scores.
    They are used for creating Figure 3 in the paper.

## License

The code is released under the terms of [Unlicense](http://unlicense.org/).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1219193.svg)](https://doi.org/10.5281/zenodo.1219193)
