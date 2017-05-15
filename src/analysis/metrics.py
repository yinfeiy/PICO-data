from pico import utils

import numpy as np
import scipy.stats as stats

def precision(spans, ref_mask):
    precision_arr = []
    for span in spans:
        length = span[1]-span[0]
        poss = sum(ref_mask[span[0]:span[1]])
        precision_arr.append(1.0*poss / length)
    precision = np.mean(precision_arr)

    return precision

def recall(gold_spans, anno_mask):
    recall_arr = []
    for span in gold_spans:
        length = span[1]-span[0]
        poss = sum(anno_mask[span[0]:span[1]])
        recall_arr.append(1.0*poss / length)
    recall = np.mean(recall_arr)

    return recall

def corr(mask, ref_mask):
    c, p = stats.spearmanr(mask, ref_mask)
    return c

def metrics(spans, ref_spans, ntokens, metric_name):
    # span mask
    m1 = [0] * ntokens
    for span in spans:
        m1[span[0]:span[1]] = [1] * (span[1]-span[0])

    # ref span mask
    m2 = [0] * ntokens
    for span in ref_spans:
        m2[span[0]:span[1]] = [1] * (span[1]-span[0])

    if metric_name == 'prec':
        score = precision(spans, m2)
    elif metric_name == 'recl':
        score = recall(ref_spans, m1)
    elif metric_name == 'corr':
        score = corr(m1, m2)

    return score

