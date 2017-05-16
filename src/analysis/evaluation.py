from pico.corpus import Corpus, Doc
from pico import utils


import numpy as np
from collections import defaultdict
import scipy.stats as stats
import metrics
import json

def evaluating_worker(corpus, annotype, metric_name):
    worker_scores = {}

    for docid in corpus.docs:
        doc = corpus.docs[docid]

        gt = doc.get_groundtruth(annotype)
        if not gt:
            continue

        markups = doc.markups[annotype]

        for wid, spans in markups.items():
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


def docs_with_gt(gt_fn):
    docids = []
    with open(gt_fn) as fin:
        for line in fin:
            item = json.loads(line.strip())
            docids.append( item['docid'] )
    return docids

if __name__ == '__main__':
    anno_path = '../annotations/'
    doc_path = '../docs/'

    anno_fn = anno_path + 'PICO-annos-crowdsourcing.json'
    gt_fn = anno_path + 'PICO-annos-professional.json'

    docids = docs_with_gt(gt_fn)

    # Loading corpus
    corpus = Corpus(doc_path = doc_path)
    corpus.load_annotations(anno_fn, docids)
    corpus.load_groudtruth(gt_fn)

    annotypes = ['Participants', 'Intervention', 'Outcome']

    worker_scores = defaultdict(dict)
    for annotype in annotypes:
        print 'Processing ', annotype
        for metric_name in ['corr', 'prec', 'recl']:
            worker_scores_annotype = evaluating_worker(corpus, annotype, metric_name)
            for wid in worker_scores_annotype:
                worker_scores[wid][metric_name] = worker_scores_annotype[wid]['score']
                worker_scores[wid]['count'] = worker_scores_annotype[wid]['count']

        with open(annotype+'.csv', 'w+') as fout:
            fout.write('workerid, numebr of doc, corr, prec, recl\n')
            for wid, scores in worker_scores.items():
                try:
                    fout.write('{0}, {1}, {2}, {3}, {4}\n'.format(wid, scores['count'], scores['corr'], scores['prec'], scores['recl']))
                except:
                    print scores
                    continue


