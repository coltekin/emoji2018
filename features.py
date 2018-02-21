import os.path
import re
import pickle, hashlib
from sklearn.externals import joblib
from logging import debug, info
from ngram import get_ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

def identity(x):
    return x

def preprocess(docs, c_ngmin=1, c_ngmax=1,
        w_ngmin=1, w_ngmax=1, lowercase=None):
    # convert docs to word/char ngrams with optional case normaliztion
    # this would ideally be tha anlyzer parameter of the
    # vectorizer, but requires lambda - which breaks saving 
    features = []
    for doc in docs:
        # character n-grams
        if lowercase == 'char':
            docfeat = get_ngrams(doc.lower(),
                    ngmax=c_ngmax, ngmin=c_ngmin,
                    tokenizer=list)
        else:
            docfeat = get_ngrams(doc,
                    ngmax=c_ngmax, ngmin=c_ngmin,
                    tokenizer=list)
        # word n-grams
        if lowercase == 'word':
            docfeat.extend(get_ngrams(doc.lower(),
                        ngmax=w_ngmax, ngmin=w_ngmin,
                        append="W"))
        else:
            docfeat.extend(get_ngrams(doc,
                        ngmax=w_ngmax, ngmin=w_ngmin,
                        append="W"))
        features.append(docfeat)
    return features

def doc_to_ngrams(docs, use_cached=True, cache=True,
                 cache_dir='.cache', **kwargs):
    """ Return bag-of-n-grams features for the give document set
    """
    param = {
        'c_ngmax': 1, 'c_ngmin': 1, 'w_ngmax': 1, 'w_ngmin': 1,
        'min_df': 1,
        'sublinear': True,
        'norm': 'l2',
        'max_features': None,
        'input_name': None,
        'lowercase': None,
        'dim_reduce': None
    }
    for k, v in kwargs.items(): param[k] = v

    if param['input_name'] and use_cached or cache:
        os.makedirs(cache_dir, exist_ok=True)
        paramstr = ','.join([k + '=' + str(param[k]) for k in sorted(param)])
        cachefn = 'vectorizer-' + \
                hashlib.sha224(paramstr.encode('utf-8')).hexdigest() + '.z'
        cachefn = os.path.join(cache_dir, cachefn)
    if use_cached and os.path.exists(cachefn):
        info('Using cached vectorizer: {}'.format(cachefn))
        fp = open(cachefn, 'r')
        v = joblib.load(cachefn)
        vectors = joblib.load(cachefn.replace('vectorizer-', 'vectors-'))
        fp.close()
    else:
        features = preprocess(docs, c_ngmin=param['c_ngmin'],
            c_ngmax=param['c_ngmax'], w_ngmin=param['w_ngmin'],
            w_ngmax=param['w_ngmax'], lowercase=param['lowercase'])
        v = TfidfVectorizer(analyzer=identity,
                            lowercase=(param['lowercase'] == 'all'),
                            sublinear_tf=param['sublinear'],
                            min_df=param['min_df'],
                            norm=param['norm'],
                            max_features=param['max_features'])
        vectors = v.fit_transform(features)
        if cache and param['input_name']:
            info('Saving vectorizer: {}'.format(cachefn))
            joblib.dump(v, cachefn, compress=True)
            joblib.dump(vectors,
                    cachefn.replace('vectorizer-', 'vectors-'),
                    compress=True)

    svd = None
    if param['dim_reduce']:
        info("reducing dimentionality {} -> {}".format(
            len(v.vocabulary_), param['dim_reduce']))
        svd = TruncatedSVD(n_components=param['dim_reduce'], n_iter=10)
#        svd = TruncatedSVD(n_components=dim_reduce, #        algorithm="arpack")
        svd.fit(vectors)
        info("explained variance: {}".format(
            svd.explained_variance_ratio_.sum()))
        vectors = svd.transform(vectors)

    return vectors, v, None


w_tokenizer = re.compile(r"\w+|[^ \t\n\r\f\v\w]+").findall


def doc_to_numseq(doc, vocab, tokenizer="char", pad=None):
    """ Transform given sequence of labels to numeric values 
    """
    from keras.preprocessing.sequence import pad_sequences
    oov_char = 1
    start_char = 2
    end_char = 3
    features = {k:v+4 for v,k in enumerate(vocab.keys())}
    X = []
    maxlen = 0
    for d in doc:
        x = [start_char]
        if tokenizer == "word":
            tokenizer = w_tokenizer
        elif tokenizer == "char":
            tokenizer = list
        tokens = tokenizer(d)
        for c in tokens:
            if c in features:
                x.append(features[c])
            else:
                x.append(oov_char)
        x.append(end_char)
        if len(x) > maxlen: maxlen = len(x)
        X.append(x)
    X = np.array(X)
    if pad:
        X = pad_sequences(X, maxlen=pad)
    return X, maxlen
