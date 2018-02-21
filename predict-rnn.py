#!/usr/bin/env python3
import sys
from emoji_data import load
from features import doc_to_numseq
import random
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers.merge import concatenate
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import SpatialDropout1D
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from logging import debug, info, basicConfig
basicConfig(level='INFO', format='%(asctime)s %(message)s')


from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument("-i", "--input", dest="input_prefix")
ap.add_argument("-o", "--output", dest="output_file", default=None)
ap.add_argument("-t", "--test", dest="test_prefix")
ap.add_argument("-b", "--batch-size", dest="batch_size", type=int,
        default=32)
ap.add_argument("-s", "--validation-ratio", dest="val_ratio",
        type=float, default=0.2)
ap.add_argument("-e", "--epoch", dest="epoch", type=int, default=10)

ap.add_argument("--c-cutoff", type=int, default = 5)
ap.add_argument("--c-maxlen", type=int, default = None)
ap.add_argument("--c-embdim", type=int, default = 64)
ap.add_argument("--c-embdrop", type=float, default = 0.2)
ap.add_argument("--c-featdim", type=int, default = 64)
ap.add_argument("--w-cutoff", type=int, default = 5)
ap.add_argument("--c-featdrop", type=float, default = 0.2)
ap.add_argument("--w-maxlen", type=int, default = None)
ap.add_argument("--w-embdim", type=int, default = 64)
ap.add_argument("--w-embdrop", type=float, default = 0.2)
ap.add_argument("--w-featdim", type=int, default = 64)
ap.add_argument("--w-featdrop", type=float, default = 0.2)
ap.add_argument("--rnn", default = 'GRU')

opt = ap.parse_args()
o = opt

data_trn = load(opt.input_prefix)
data_tst = load(opt.test_prefix)

trn_labels = to_categorical(np.array(data_trn.labels))

if not o.c_maxlen:
    o.c_maxlen = np.max(data_trn.len_char)

c_vocab = Counter({k:v for k,v in data_trn.chars.items() if v > o.c_cutoff})
c_trn, _ = doc_to_numseq(np.array(data_trn.docs), vocab=c_vocab,
        pad=o.c_maxlen)
c_tst, _ = doc_to_numseq(np.array(data_tst.docs), vocab=c_vocab,
        pad=o.c_maxlen)

if not o.w_maxlen:
    o.w_maxlen = np.max(data_trn.len_word)

w_vocab =  Counter({k:v for k,v in data_trn.words.items() if v > o.w_cutoff})
w_trn, _ = doc_to_numseq(np.array(data_trn.docs), vocab=w_vocab,
        tokenizer="word", pad=o.w_maxlen)
w_tst, _ = doc_to_numseq(np.array(data_tst.docs), vocab=w_vocab,
        tokenizer="word", pad=o.w_maxlen)

c_inp = Input(shape=(o.c_maxlen, ), name='char_input')
w_inp = Input(shape=(o.w_maxlen, ), name='word_input')

c_emb = Embedding(len(c_vocab) + 4, o.c_embdim, mask_zero=True,
        name='char_embedding')(c_inp)
c_emb = SpatialDropout1D(o.c_embdrop)(c_emb)
w_emb = Embedding(len(w_vocab) + 4, o.w_embdim, mask_zero=True,
        name='word_embedding')(w_inp)
w_emb = SpatialDropout1D(o.w_embdrop)(w_emb)

if o.rnn == 'LSTM':
    rnn = LSTM
else:
    rnn = GRU

c_fw = rnn(o.c_featdim, dropout=o.c_featdrop, name='char_fw_rnn')(c_emb)
c_bw = rnn(o.c_featdim, dropout=o.c_featdrop, go_backwards=True,
        name='char_bw_rnn')(c_emb)
c_feat = concatenate([c_fw, c_bw])

w_fw = rnn(o.w_featdim, dropout=o.w_featdrop, name='word_fw_rnn')(w_emb)
w_bw = rnn(o.w_featdim, dropout=o.w_featdrop, go_backwards=True,
        name='word_bw_rnn')(w_emb)
w_feat = concatenate([w_fw, w_bw])

h = concatenate([c_feat, w_feat])

emo = Dense(trn_labels.shape[1], activation='softmax', name='emoji')(h)

m = Model(inputs=[c_inp, w_inp], outputs=[emo])
#    m.summary()

m.compile(loss='categorical_crossentropy', optimizer='adam',
        metrics=['accuracy'])

m.fit(x={'char_input': c_trn, 'word_input': w_trn},
      y=trn_labels,
      batch_size=opt.batch_size,
      epochs=opt.epoch)

pred = np.argmax(
        m.predict(x={'char_input': c_tst, 'word_input':
            w_tst},batch_size=opt.batch_size),
        axis=1)

outfp = sys.stdout
if opt.output_file:
    outfp = open(opt.output_file, "w")

for lab in pred:
    print(lab, file=outfp)

if opt.output_file:
    outfp.close()
