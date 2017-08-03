import os, sys
import json
import pprint
from collections import defaultdict
import matplotlib.pyplot as plt

plt.switch_backend('agg')

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

ifn = './difficulty_annotated.json'

annotypes = ['Participants', 'Intervention', 'Outcome']
GT_TEMPLATE = '{0}_gt_mask'

gts = defaultdict(list)
dts = defaultdict(list)

with open(ifn) as fin:
    for line in fin:
        doc = json.loads(line)
        parsed_text = doc['parsed_text']

        for sent in parsed_text['sents']:
            for annotype in annotypes:
                dt_key = '{0}_prob'.format(annotype)
                gt_key = GT_TEMPLATE.format(annotype)

                if gt_key not in sent:
                    continue

                gt = 1 if sum(sent[gt_key]) > 0 else 0
                dt = sent[dt_key]
                gts[annotype].append(gt)
                dts[annotype].append(dt)

precision = {}; recall = {}; average_precision = {}
for annotype in annotypes:
    precision[annotype], recall[annotype], _ = precision_recall_curve(gts[annotype], dts[annotype])
    average_precision[annotype] = average_precision_score(gts[annotype], dts[annotype])
    print average_precision[annotype]

# Plot Precision-Recall curve
colors = dict(Participants='navy', Intervention='turquoise', Outcome='darkorange')

plt.clf()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall PCO'.format())
for annotype in annotypes:
    plt.plot(recall[annotype], precision[annotype], color=colors[annotype], lw=2,
            label='Precision-recall curve of class {0} (area = {1:0.2f})'.format(annotype, average_precision[annotype]))
plt.legend(loc="lower left")
plt.savefig("PICO_PR_curve_gt.png")
