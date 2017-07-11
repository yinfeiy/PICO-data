from pico.corpus import Corpus, Doc
from pico import utils

import numpy as np
from collections import defaultdict
import scipy.stats as stats
import matplotlib.pyplot as plt
import metrics
import json

def evaluating_worker_per_doc(docs, annotype, metric_name):

    worker_scores = {}

    for docid in docs:
        doc = docs[docid]

        gt = doc.get_groundtruth(annotype)
        if not gt or annotype not in doc.markups:
            if annotype not in doc.markups:
                # print docid
                pass
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

if __name__ == '__main__':
    doc_path = '../docs/'

    annotypes = ['Participants', 'Intervention', 'Outcome']
    anno_fn = '/mnt/data/workspace/nlp/PICO-data/src/analysis/htmls/output/tmp_min6.json'
    #anno_fn = '../annotations/PICO-annos-crowdsourcing.json'

    gt_fn = '../annotations/PICO-annos-professional.json'
    #gt_wids = ['AXQIZSZFYCA8T']
    #gt_wids = ['md2']
    gt_wids = None

    docids = utils.docs_with_gt(gt_fn)

    # Loading corpus
    corpus = Corpus(doc_path = doc_path, verbose=False)
    corpus.load_annotations(anno_fn, docids)
    corpus.load_groundtruth(gt_fn, gt_wids) # It will load all annotators if wid is None

    display_name = dict(mv='Majority Vote', dw='Dawid Skene', HMMCrowd='HMMCrowd')
    for annotype in annotypes:
        worker_scores = defaultdict(dict)
        print 'Processing ', annotype
        for metric_name in ['corr', 'prec', 'recl', 'f1']:
            worker_scores_annotype = evaluating_worker(corpus, annotype, metric_name)
            for wid in worker_scores_annotype:
                worker_scores[wid][metric_name] = worker_scores_annotype[wid]['score']
                worker_scores[wid]['count'] = worker_scores_annotype[wid]['count']

        scores = []
        for wid in worker_scores.keys():
            print display_name.get(wid, wid) + '\t',
            for metric_name in ['prec', 'recl', 'f1', 'corr']:
                print '& {:.3f} '.format(worker_scores[wid].get(metric_name,-1)),
                if metric_name in worker_scores[wid]:
                    scores.append(worker_scores[wid][metric_name])
            print '\\\\'

        # Plot worker score histogram
        #plt.clf()
        #plt.hist(scores, bins=20, alpha=0.5, edgecolor='w')
        #plt.title(annotype, fontsize=26)
        #plt.xlabel("worker scores", fontsize=20)
        #plt.ylabel("number of workers", fontsize=20)
        #plt.xlim([-0.2,1])
        #plt.savefig('hist_worker_scores_gt_{0}.png'.format(annotype.lower()))
        #plt.show()

        #with open(annotype+'.csv', 'w+') as fout:
        #    fout.write('workerid, numebr of doc, corr, prec, recl\n')
        #    for wid, scores in worker_scores.items():
        #        try:
        #            fout.write('{0}, {1}, {2}, {3}, {4}\n'.format(wid, scores['count'], scores['corr'], scores['prec'], scores['recl']))
        #        except:
        #            print scores
        #            continue


