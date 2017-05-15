from pico.corpus import Corpus, Doc
from pico import utils

import numpy as np
import scipy.stats as stats
import metrics

def evaluating_worker(corpus, annotype):
    worker_scores = {}

    for docid in corpus.docs:
        doc = corpus.docs[docid]

        gt = doc.get_groundtruth(annotype)
        if not gt:
            continue

        markups = doc.markups[annotype]

        for wid, spans in markups.items():
            print metrics.metrics(spans, gt, doc.ntokens, 'corr')


def docs_with_gt(gt_fn):
    # TODO
    return []

if __name__ == '__main__':
    anno_path = '../annotations/'
    doc_path = '../docs/'

    gt_fn = anno_path + 'PICO-annos-crowdsourcing.json'
    anno_fn = anno_path + 'PICO-annos-professional.json'

    docids = docs_with_gt(gt_fn)

    # Loading corpus
    corpus = Corpus(doc_path = doc_path)
    corpus.load_annotations(anno_fn, docids)
    corpus.load_groudtruth(gt_fn)

    evaluating_worker(corpus, 'Participants')


