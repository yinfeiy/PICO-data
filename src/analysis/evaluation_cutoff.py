from pico.corpus import Corpus, Doc
from pico import utils

import numpy as np
from collections import defaultdict
import scipy.stats as stats
import metrics
import json

def evaluating_worker_per_doc(docs, annotype, metric_name):

    worker_scores = {}

    for docid in docs:
        doc = docs[docid]

        gt = doc.get_groundtruth(annotype)
        if not gt or annotype not in doc.markups:
            if annotype not in doc.markups:
                print docid
            continue

        markups = doc.markups[annotype]

        for wid, spans in markups.items():
            if len(spans) == 0 : # The worker has no annotation for this doc
                continue

            score = metrics.metrics(spans, gt, doc.ntokens, metric_name)
            if np.isnan(score):
                continue

            if wid in worker_scores:
                worker_scores[wid].append(score)
            else:
                worker_scores[wid] = [score]

    for wid in worker_scores:
        worker_scores[wid] = dict(count=len(worker_scores[wid]), score=np.mean(worker_scores[wid]))

    return worker_scores


def evaluating_worker(corpus, annotype, metric_name):
    worker_scores = evaluating_worker_per_doc(corpus.docs, annotype, metric_name)
    return worker_scores


def docs_with_gt(gt_fn, annotype=None):
    docids = []
    with open(gt_fn) as fin:
        for line in fin:
            item = json.loads(line.strip())
            if annotype != None:
                gt_anno = item.get(annotype, {})
                if gt_anno:
                    docids.append( item['docid'] )
            else:
                docids.append( item['docid'] )
    return docids

def docs_anno_stats(anno_fn, pruned_workers=defaultdict(list)):
    doc_counts = defaultdict(dict)

    if not isinstance(pruned_workers, defaultdict):
        pruned_workers = defaultdict(list, pruned_workers)

    with open(anno_fn) as fin:
        for line in fin:
            annos = json.loads(line.strip())
            docid = annos['docid']

            for annotype in annos:
                if annotype == 'docid':
                    continue
                markups  = annos.get(annotype, {})
                wids = [ wid for wid in markups.keys() if wid not in pruned_workers[annotype] ]
                doc_counts[docid][annotype] = len(wids)

    return doc_counts

if __name__ == '__main__':
    doc_path = '../docs/'

    anno_fn = '/mnt/data/workspace/nlp/dawid_skene_pico/aggregated_results/PICO-annos-dw_HMM_Crowd_max_10.json'

    raw_anno_fn = '../annotations/PICO-annos-crowdsourcing.json'
    gt_fn = '../annotations/PICO-annos-professional.json'
    #gt_wids = ['AXQIZSZFYCA8T']
    #gt_wids = ['md2']
    gt_wids = None
    cutoff = 0

    doc_counts = docs_anno_stats(raw_anno_fn)

    annotypes = ['Participants', 'Intervention', 'Outcome']

    for annotype in annotypes:
        docids_with_nw = set([did for did in doc_counts if \
                doc_counts[did].get(annotype, 0) >= cutoff])
        docids_with_gt = set(docs_with_gt(gt_fn, annotype))

        docids = list(docids_with_gt.intersection(docids_with_nw))

        # Loading corpus for each annotype as number of workers are different for annotypes
        corpus = Corpus(doc_path = doc_path, verbose = False)
        corpus.load_annotations(anno_fn, docids)
        corpus.load_groundtruth(gt_fn, gt_wids) # It will load all annotators if wid is None

        print 'Processing ', annotype
        worker_scores = defaultdict(dict)
        for metric_name in ['corr', 'prec', 'recl']:
            worker_scores_annotype = evaluating_worker(corpus, annotype, metric_name)
            for wid in worker_scores_annotype:
                worker_scores[wid][metric_name] = worker_scores_annotype[wid]['score']
                worker_scores[wid]['count'] = worker_scores_annotype[wid]['count']

        print worker_scores


