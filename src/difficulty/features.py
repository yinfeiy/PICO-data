import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_extraction import DictVectorizer

def loadVocab(fname, min_df=0):
    vocab = {}
    with open(fname) as fin:
        for line in fin:
            try:
                word, df = line.strip().split(',')
            except:
                continue
            df = float(df)
            if df < min_df:
                break
            vocab[word] = df
    return vocab

def extractBOWtFeature(docs, docids, vocab, binary=True, lower=True):
    # Normalize the feature vector order
    vocab_words = vocab.keys()
    vocab_words.sort()

    feats = []
    for docid in docids:
        doc = docs[docid]
        text = doc['text']
        if lower:
            text = text.lower()
        tokens = text.split(' ')
        num = len(tokens)*1.0

        token_count = defaultdict(int)
        for token in tokens:
            if token in vocab:
                token_count[token] += 1

        if binary:
            feat = [1 if token in token_count else 0 for token in vocab_words]
        else:
            feat = [token_count[token]/num if token in token_count else 0 for token in vocab_words]
        feats.append(feat)

    return np.array(feats)

def extractMetaFeature(docs, docids):

    feats = []
    for docid in docids:
        doc = docs[docid]

        feat = []
        text_arr = doc['text'].split()
        pos_arr = doc['pos'].split()

        # Length
        num = len(text_arr)

        # number of NUMs, PUNCTs, unicode tokens
        cnt_num = 0; cnt_punct = 0; cnt_uni = 0
        for pos in pos_arr:
            if pos == 'NUM':
                cnt_num += 1
            elif pos == 'PUNCT':
                cnt_punct += 1

        for token in text_arr:
            try:
                token.encode('ascii')
            except:
                cnt_uni += 1

        feat = [num, cnt_num, cnt_punct, cnt_uni]
        feats.append(feat)
    return feats
