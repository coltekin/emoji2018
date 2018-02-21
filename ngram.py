#!/usr/bin/env python3

import re

# simple tokenizer to extract sequences of letters, or non-space
# single symbols
w_tokenizer = re.compile("\w+|[^ \t\n\r\f\v\w]+")

def get_ngrams(s, ngmin=1, ngmax=1,
                tokenizer=w_tokenizer.findall,
                separator="|",
                bos="<",
                eos=">",
                append="",
                flatten=True):
    """ Return all ngrams with in range ngmin-ngmax.
        The function specified by 'tokenizer' should return a list of
        tokens. By default, the tokenizer splits the string into word
        and non-word non-space characters.
    """
    ngrams = [[] for x in range(1, ngmax + 1)]
    s = tokenizer(bos + s + eos)
    for i, ch in enumerate(s):
        for ngsize in range(ngmin, ngmax + 1):
            if (i + ngsize) <= len(s):
                ngrams[ngsize - 1].append(separator.join(s[i:i+ngsize]))
    if flatten:
        ngrams = [ng for nglist in ngrams for ng in nglist]
    return ngrams

if __name__ == '__main__':
    print(get_ngrams("abc, xyz...klm abc-def mmx"))
    print(get_ngrams("abç, ğyz...klm 汉语 mmx"))
    print(get_ngrams("abc, xyz...klm mmx", tokenizer=list))
    print(get_ngrams("abc, xyz...klm mmx", 1, 2))
    print(get_ngrams("abc, xyz...klm mmx", 1, 2,flatten=False))
