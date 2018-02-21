def add_args(ap, groups):
    """ Add command line arguments to an already initialized
        ArgumentParser object.
        'group' allows adding particular arguments useful for for
        in particular appalications.
    """

    if 'general' in groups:
        ap.add_argument("-i", "--input", dest="input_prefix")
        ap.add_argument("-D", "--cach-dir", dest="cache_dir",
                default=".cache")
        ap.add_argument("-l", "--log-level", dest="log_level",
                default="INFO")
        ap.add_argument("-j", "--n-jobs", dest="n_jobs", type=int,
                default=-1)
    if 'preproc' in groups: # pre-processing options
        ap.add_argument("-f", "--min-df", dest="min_df", type=int,
                default=1)
        ap.add_argument("-c", "--min-char-ng", dest="c_ngmin",
                type=int, default=1)
        ap.add_argument("-w", "--min-word-ng", dest="w_ngmin",
                type=int, default=1)
        ap.add_argument("-C", "--max-char-ng", dest="c_ngmax",
                type=int, default=1)
        ap.add_argument("-W", "--max-word-ng", dest="w_ngmax",
                type=int, default=1)
        ap.add_argument("-d", "--reduce-dim", dest="dim_reduce",
                type=int, default=None)
        ap.add_argument("-L", "--lowercase", dest="lowercase",
                default=None)
    if 'linear' in groups: # options relevant for linear models
        ap.add_argument("-r", "--regularizatio-constant",
                dest="C", type=float, default=1.0)
        ap.add_argument("--no-class-weight", dest="class_weight",
                action="store_false")
        ap.add_argument("-M", "--multi-class", dest="mult_class",
                default='ovr')
        ap.add_argument("-m", "--classifier", dest="classifier",
                default='SVM')
    if 'tune' in groups: # options relevant during tuning
        ap.add_argument("-k", "--folds", dest="k", type=int,
                default=5)
    if 'test' in groups: # options relevant during testing
        ap.add_argument("-t", "--test", dest="test_prefix")
