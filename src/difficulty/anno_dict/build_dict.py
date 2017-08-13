import os, sys
import json
import numpy as np
from collections import defaultdict

ifn = '../difficulty_annotated.json'

def isTest(doc):
    for key in doc:
        if key.endswith('corr_gt') and doc[key]:
            return True
    return False

def extractMarkedTokens(doc, annotype, vocab_dict):
    parsed_text = doc['parsed_text']
    for sent in parsed_text['sents']:
        key = '{0}_mv_mask'.format(annotype)
        idxs = np.where(np.array(sent[key]) > 0)
        for idx in idxs[0]:
            word, pos_1, pos_2 = sent['tokens'][idx]
            try:
                word.encode('ascii')
            except:
                continue

            if pos_1 in ['PUNCT', 'NUM']: # Also, speical characters are important
                continue
            vocab_dict[word.lower()] += 1

annotypes = ['Participants', 'Intervention', 'Outcome']

at_dict = {}
for at in annotypes:
    at_dict[at] = defaultdict(int)

with open(ifn) as fin:
    for line in fin:
        item = json.loads(line)
        if not isTest(item):
            for at in annotypes:
                extractMarkedTokens(item, at, at_dict[at])

for at in annotypes:
    vocab_dict=at_dict[at]
    words = vocab_dict.keys()
    #words = [w for w in words if vocab_dict[w] > 1]
    words.sort(key=lambda x:vocab_dict[x], reverse=True)

    ofn = '{0}_vocab.dict'.format(at)
    with open(ofn, 'w+') as fout:
        for word in words:
            fout.write('{0}, {1}\n'.format(word, vocab_dict[word]))
