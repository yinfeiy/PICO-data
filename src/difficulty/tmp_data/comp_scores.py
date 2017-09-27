import json

f1 = 'difficulty.json'
f2 = 'difficulty_weighted.json'

f1_scores = {}
with open(f1) as fin:
    for line in fin:
        item = json.loads(line)
        id = item['docid']
        val = item['Outcome_recl_gt']
        f1_scores[id] = val

f2_scores = {}
with open(f2) as fin:
    for line in fin:
        item = json.loads(line)
        id = item['docid']
        val = item['Outcome_recl_gt']
        f2_scores[id] = val

for id in f1_scores:
    print f1_scores[id], f2_scores[id]
