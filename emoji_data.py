""" Read emoji prediction shared task documents
"""

from collections import Counter
import os.path
import re

def load(prefix):
    return EmojiSharedTaskData(prefix)

class EmojiSharedTaskData():
    __slots__ = ('docids', 'docs', 
                 'labels', 'labelchar', 'labelname',
                 'len_char', 'len_word', 'words', 'chars', 'tokenizer')

    def __init__(self, prefix=None,
               tokenizer=re.compile("\w+|[^ \t\n\r\f\v\w]+").findall):
        self.docids = []
        self.docs = []
        self.labels = []
        self.labelchar = []
        self.labelname = []
        self.len_char = []
        self.len_word = []
        self.words = Counter()
        self.chars = Counter()
        self.tokenizer = tokenizer
        if prefix is not None:
            self.load(prefix)

    def load(self, prefix):
        with open(prefix + '.text', 'r') as fp:
            for doc in fp:
                doc = doc.strip()
                self.docs.append(doc)
                self.len_char.append(len(doc))
                self.chars.update(list(doc))
                if self.tokenizer:
                    tokens = self.tokenizer(doc)
                    self.len_word.append(len(tokens))
                    self.words.update(tokens)
        self.docids = list(range(len(self.docs)))
        if os.path.exists(prefix + '.labels'):
            with open(prefix + '.labels', 'r') as fp:
                self.labels = [lab.strip() for lab in fp.readlines()]
        if "us_" in prefix:
            mapping = os.path.join(os.path.dirname(prefix), "us_mapping.txt")
        else:
            mapping = os.path.join(os.path.dirname(prefix), "es_mapping.txt")
        if os.path.exists(mapping):
            with open(mapping, 'r') as fp:
                for line in fp:
                    _, lab, labname = line.strip().split()
                    self.labelchar.append(lab)
                    self.labelname.append(labname)
