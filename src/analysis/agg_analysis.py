""" This file is for analysising if calculating worker score regarding agreegated results is a good reference of difficulty"""

from pico.corpus import Corpus, Doc
from pico import utils

import numpy as np
from collections import defaultdict
import scipy.stats as stats
import matplotlib.pyplot as plt
import metrics
import json

def evaluating_doc_scores(docs, annotype, metric_name):

    doc_scores = {}

    for docid in docs:
        doc = docs[docid]

        gt = doc.get_groundtruth(annotype)
        if not gt or annotype not in doc.markups:
            if annotype not in doc.markups:
                # print docid
                pass
            continue

        markups = doc.markups[annotype]

        scores = []
        for wid, spans in markups.items():
            if len(spans) == 0 : # The worker has no annotation for this doc
                continue

            score = metrics.metrics(spans, gt, doc.ntokens, metric_name)
            if np.isnan(score):
                continue

            scores.append(score)
        doc_scores[docid] = np.mean(scores)

    return doc_scores


def evaluating_doc(corpus, annotype, metric_name):
    doc_scores = evaluating_doc_scores(corpus.docs, annotype, metric_name)
    return doc_scores

def process(corpus):
    pass

if __name__ == '__main__':
    doc_path = '../docs/'

    annotypes = ['Participants', 'Intervention', 'Outcome']
    anno_fn = '../annotations/PICO-annos-crowdsourcing.json'

    gt_fn_1 = '../annotations/PICO-annos-professional.json'
    gt_wids_1 = None
    docids = utils.docs_with_gt(gt_fn_1)

    gt_fn_2 = '../annotations/PICO-annos-crowdsourcing-agg.json'
    gt_wids_2 = None

    # Loading corpus
    corpus_1 = Corpus(doc_path = doc_path, verbose=False)
    corpus_1.load_annotations(anno_fn, docids)
    corpus_1.load_groundtruth(gt_fn_1, gt_wids_1) # It will load all annotators if wid is None

    corpus_2 = Corpus(doc_path = doc_path, verbose=False)
    corpus_2.load_annotations(anno_fn, docids)
    corpus_2.load_groundtruth(gt_fn_2, gt_wids_2)

    display_name = dict(mv='Majority Vote', dw='Dawid Skene', HMMCrowd='HMMCrowd')
    for annotype in annotypes:
        print 'Processing ', annotype
        doc_scores_1 = defaultdict(dict)
        for metric_name in ['corr', 'prec', 'recl', 'f1']:
            doc_scores_annotype = evaluating_doc(corpus_1, annotype, metric_name)
            for docid in doc_scores_annotype:
                doc_scores_1[docid][metric_name] = doc_scores_annotype[docid]

        doc_scores_2 = defaultdict(dict)
        for metric_name in ['corr', 'prec', 'recl', 'f1']:
            doc_scores_annotype = evaluating_doc(corpus_2, annotype, metric_name)
            for docid in doc_scores_annotype:
                doc_scores_2[docid][metric_name] = doc_scores_annotype[docid]

        for metric_name in ['corr', 'prec', 'recl', 'f1']:
            print metric_name, ':',
            ss1=[];  ss2=[]
            for docid in doc_scores_1.keys():
                if docid not in doc_scores_2:
                    continue

                s1 = doc_scores_1[docid][metric_name]
                s2 = doc_scores_2[docid][metric_name]
                if np.isnan(s1) or np.isnan(s2):
                    continue
                ss1.append(s1)
                ss2.append(s2)

            pr, _= stats.pearsonr(ss1, ss2)
            sr, _= stats.spearmanr(ss1, ss2)
            print '{0:.3f} / {1:.3f},\t'.format(pr, sr),
        print ''


