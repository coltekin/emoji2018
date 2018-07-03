#!/usr/bin/env python3
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
opt = ap.parse_args()

class Options:
    __slots__ = (
        'c_cutoff',
        'c_maxlen',
        'c_embdim',
        'c_embdrop',
        'c_featdim',
        'w_cutoff',
        'c_featdrop',
        'w_maxlen',
        'w_embdim',
        'w_embdrop',
        'w_featdim',
        'w_featdrop',
        'rnn')
    def __init__(self,
            c_cutoff = 5,
            c_maxlen = None,
            c_embdim = 64,
            c_embdrop = 0.2,
            c_featdim = 64,
            w_cutoff = 5,
            c_featdrop = 0.2,
            w_maxlen = None,
            w_embdim = 64,
            w_embdrop = 0.2,
            w_featdim = 64,
            w_featdrop = 0.2,
            rnn = 'GRU'):
        self.c_cutoff = c_cutoff
        self.c_maxlen = c_maxlen
        self.c_embdim = c_embdim
        self.c_embdrop = c_embdrop
        self.c_featdim = c_featdim
        self.w_cutoff = w_cutoff
        self.c_featdrop = c_featdrop
        self.w_maxlen = w_maxlen
        self.w_embdim = w_embdim
        self.w_embdrop = w_embdrop
        self.w_featdim = w_featdim 
        self.w_featdrop = w_featdrop 
        self.rnn = rnn  

    @classmethod
    def sample(cls):
        o = Options(
            c_cutoff = random.choice((1, 4)),
            c_maxlen = None,
            c_embdim = random.choice((32, 64)),
            c_embdrop = random.choice((0.1, 0.2, 0.5)),
            c_featdim = random.choice((32, 64, 128)),
            w_cutoff =  random.choice((1, 4)),
            c_featdrop = random.choice((64, 128, 256)),
            w_maxlen = None,
            w_embdim = random.choice((32, 64, 128)),
            w_embdrop = random.choice((0.1, 0.2, 0.5)),
            w_featdim = random.choice((64, 128, 256)),
            w_featdrop = random.choice((0.1, 0.2, 0.5)),
            rnn = random.choice(('GRU', 'LSTM')))
        
        return o, hash(str(o))
    def __str__(self):
        str = ""
        for attr in self.__slots__:
            str += '{}={}, '.format(attr, getattr(self, attr))
        return str[:-2]

data = load(opt.input_prefix)
nepoch = 40

ssp = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
ssp.get_n_splits(data.docs, data.labels)
trn_idx, dev_idx = list(ssp.split(data.docs, data.labels))[0]

trn_labels = to_categorical(np.array(data.labels)[trn_idx])
dev_labels = to_categorical(np.array(data.labels)[dev_idx])

search_iter = 1000
search_done = set()
for _ in range(search_iter):
    o, h = Options.sample()
    if h in search_done: continue
    search_done.add(h)

    if not o.c_maxlen:
        o.c_maxlen = np.max(data.len_char)
    c_vocab = Counter({k:v for k,v in data.chars.items() if v > o.c_cutoff})
    c_trn, _ = doc_to_numseq(np.array(data.docs)[trn_idx], vocab=c_vocab,
            pad=o.c_maxlen)
    c_dev, _ = doc_to_numseq(np.array(data.docs)[dev_idx], vocab=c_vocab,
            pad=o.c_maxlen)

    if not o.w_maxlen:
        o.w_maxlen = np.max(data.len_word)
    w_vocab =  Counter({k:v for k,v in data.words.items() if v > o.w_cutoff})
    w_trn, _ = doc_to_numseq(np.array(data.docs)[trn_idx], vocab=w_vocab,
            tokenizer="word", pad=o.w_maxlen)
    w_dev, _ = doc_to_numseq(np.array(data.docs)[dev_idx], vocab=w_vocab,
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
    m.summary()

    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    class Metrics(Callback):
        def on_train_begin(self, logs={}):
            self.val_f1 = []
            self.val_precision = []
            self.val_recall = []
            self.val_accuracy = []
        def on_epoch_end(self, batch, logs={}):
            predict = np.argmax(self.model.predict(self.validation_data[:2]), axis=1)
            targ = np.argmax(self.validation_data[2], axis=1)
            prec, rec, f1, _ = precision_recall_fscore_support(targ, predict, average='macro')
            self.val_f1.append(f1)
            self.val_precision.append(prec)
            self.val_recall.append(rec)
            self.val_accuracy.append(accuracy_score(targ, predict))
        def __str__(self):
            return 'prfa: {} {} {} {}'.format(
                    self.val_precision, self.val_recall, self.val_f1,
                    self.val_accuracy)


    prf_scores = Metrics()

    early_stop = EarlyStopping(monitor='val_loss', patience=4)

    info("Fitting: {}".format(str(o)))
    m.fit(x={'char_input': c_trn, 'word_input': w_trn},
          y=trn_labels,
          validation_data=({'char_input': c_dev, 'word_input': w_dev}, dev_labels),
          epochs=nepoch, callbacks=[prf_scores, early_stop], verbose=0)
    info("Results: {}".format(str(prf_scores)))
