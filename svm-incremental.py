#!/usr/bin/env python3

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import numpy as np

from emoji_data import load
from features import doc_to_ngrams

from cmdline import add_args
from argparse import ArgumentParser
ap = ArgumentParser()
add_args(ap, ('general', 'preproc', 'linear', 'tune'))
opt = ap.parse_args()

seed=1234

if opt.class_weight:
    opt.class_weight = "balanced"
else:
    opt.class_weight = None

from logging import debug, info, basicConfig
basicConfig(level=opt.log_level,
                    format='%(asctime)s %(message)s')

info('----start----')
info(','.join([k + '=' + str(vars(opt)[k]) for k in sorted(vars(opt))]))


# ---main---

data = load(opt.input_prefix)

docs, v, _ = doc_to_ngrams(data.docs, min_df=opt.min_df,
                          cache_dir = opt.cache_dir,
                          dim_reduce = opt.dim_reduce,
                          c_ngmin = opt.c_ngmin,
                          c_ngmax = opt.c_ngmax,
                          w_ngmin = opt.w_ngmin,
                          w_ngmax = opt.w_ngmax,
                          lowercase = opt.lowercase,
                          input_name = opt.input_prefix)
labels = np.array(data.labels)

info("number of word/character features ({}/{}): {}".format(
            opt.w_ngmax, opt.c_ngmax, len(v.vocabulary_)))

if opt.classifier == 'lr': 
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(dual=True, C=opt.C, verbose=0,
            class_weight=opt.class_weight)
elif opt.classifier == 'rf':
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(class_weight=opt.class_weight,n_estimators=300,random_state=seed)
else:
    from sklearn.svm import LinearSVC
    m = LinearSVC(dual=True, C=opt.C, verbose=0,
            class_weight=opt.class_weight)
   
if opt.mult_class == 'ovo':
    mc = OneVsOneClassifier
elif opt.mult_class == 'ovr':
    mc = OneVsRestClassifier
if opt.classifier != 'rf':
    m = mc(m, n_jobs=opt.n_jobs)

ssp = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
ssp.get_n_splits(docs, labels)
trn_idx, dev_idx = list(ssp.split(data.docs, data.labels))[0]

acc = []
f1M = []

train_docs = docs[trn_idx]
train_labels = labels[trn_idx]
dev_docs = docs[dev_idx]
dev_labels = labels[dev_idx]
split_size = round(len(trn_idx)/10)
for i in range(10):
    info("training up to {}".format((i+1)*split_size))
    m.fit(train_docs[0:(i+1)*split_size], train_labels[0:(i+1)*split_size])
    pred = m.predict(dev_docs)
    acc.append(accuracy_score(dev_labels, pred))
    f1M.append(f1_score(dev_labels, pred, average='macro'))

info("Accuracy: {}.".format(acc))
info("F1(macro): {}.".format(f1M))

info('----end----')

# ll = sorted(set(data.labels), key=lambda x: int(x))
# fmt = "{:>3}" + "{:>7}" * (len(ll))
# print(fmt.format(" ", *data.labelchar))
# for i, row in enumerate(confusion_matrix(data.labels, pred, data.labels=ll)):
#     print(fmt.format(data.labelchar[i], *row))
