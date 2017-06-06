import hmm, util
import json
from collections import defaultdict
import sys
sys.path.insert(0, '/mnt/data/workspace/nlp/PICO-data/src/')

import pico_data

def list_word_spans(x):
    res = []
    i = 0
    while i < len(x):
        if (i < len(x)) and (x[i] == 0): i += 1; continue
        start = i
        while (i < len(x)) and (x[i] == 1): i += 1
        end = i - 1
        res.append( (start, end))
    return res

def get_pid(ins):
    s = inv_l[ins.label]
    a = map(int,s.split('_'))
    return a[0]

def get_start(ins):
    s = inv_l[ins.label]
    a = map(int,s.split('_'))
    return a[1]

def get_end(ins):
    s = inv_l[ins.label]
    a = map(int,s.split('_'))
    return a[2]

annotype  = 'Outcome'
if len(sys.argv) > 1:
    annotype = sys.argv[1]

print annotype

init_type = 'dw'
max_num_worker = 10

#pico.read_ann()
#(cd, gold, list_wid, features, labels) = pico.main()
(cd, list_wid, features, labels) = pico_data.main(annotype, max_num_worker=max_num_worker)

n = 2
m = len(features) + 1
hc = hmm.HMM_crowd(n, m, cd, features, labels, n_workers=len(list_wid), ne = 0, smooth = 1e-3)

hc.init(init_type=init_type, wm_rep='cm', dw_em = 5, wm_smooth=0.001)

inv_l = {v:k for (k,v) in labels.items()}

# Results in a before em step
results_init = defaultdict(list)

for s, r in zip(hc.data.sentences, hc.d.res):
    if len(s) == 0: continue
    pid = get_pid(s[0])
    spans = list_word_spans(r)
    for l,r in spans:
        start = get_start(s[l])
        end = get_end(s[r])
        results_init[pid].append([start, end])

with open('aggregated_results/' + annotype + '-aggregated_{0}_max_{1}.json'.format(init_type, max_num_worker), 'w+') as fo:
    for pid in results_init:
        annos = results_init[pid]
        item = {"docid":pid, annotype:annos}
        fo.write(json.dumps(item)+'\n')

hc.vb = [0.1, 0.1]
hc.em(5)
hc.mls()

results = defaultdict(list)
for s, r in zip(hc.data.sentences, hc.res):
    if len(s) == 0: continue
    pid = get_pid(s[0])
    spans = list_word_spans(r)
    for l,r in spans:
        start = get_start(s[l])
        end = get_end(s[r])
        results[pid].append([start, end])

with open('aggregated_results/' + annotype + '-aggregated_{0}_HMM_Crowd_max_{1}.json'.format(init_type, max_num_worker), 'w+') as fo:
    for pid in results:
        annos = results[pid]
        item = {"docid":pid, annotype:annos}
        fo.write(json.dumps(item)+'\n')


