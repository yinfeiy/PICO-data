import os, sys
from collections import defaultdict

fn = 'subtitles.txt'

words = defaultdict(int)
with open(fn) as fin:
    cnt = 0
    for line in fin:
        cnt += 1
        for t in line.strip().split():
            words[t] += 1
        if cnt % 100000 == 0:
            print cnt

word_list = words.keys()
word_list.sort(key=lambda x:words[x], reverse=True)

with open('vocab.txt', 'w+') as fout:
    for w in word_list:
        fout.write('{0}\t{1}\n'.format(w, words[w]))
