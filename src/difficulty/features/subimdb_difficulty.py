import os, sys

DICT_FILE='/mnt/data/workspace/nlp/PICO-data/src/difficulty/features/subimdb/vocab.txt'

def _build_vocab():
    res = {}
    with open(DICT_FILE) as fin:
        for line in fin:
            word, freq = line.strip().split('\t')
            res[word] = freq
    return res

vocab = _build_vocab()

if __name__ == '__main__':
    print len(vocab)
