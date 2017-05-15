from pico.corpus import Corpus, Doc
from pico import utils

import numpy as np
import scipy.stats as stats

DOC_PATH = '../docs/'

def worker_scores_doc_corr(doc, annotype, pruned_workers):
    # Leave One Out
    markups = doc.markups[annotype]

    workers = [w for w in markups.keys() if w not in pruned_workers]
    nworker = len(workers)

    markup_mask = np.zeros(doc.ntokens)
    for i in range(nworker):
        spans = markups[workers[i]]
        for span in spans:
            markup_mask[span[0]:span[1]] = markup_mask[span[0]:span[1]] + [1] * (span[1]-span[0])

    worker_scores = {}
    for i in range(nworker):
        worker_mask = np.zeros(doc.ntokens)
        spans = markups[workers[i]]
        for span in spans:
            worker_mask[span[0]:span[1]] = [1] * (span[1]-span[0])

        if nworker == 1:
            print "[Warn] Only one worker for doc {0}, do not calculate worker score.".format(doc.docid)
            continue

        elif len(worker_mask) == sum(worker_mask):
            c = 0
        else:
            mask_loo = (markup_mask - worker_mask) / (nworker-1)
            c, p = stats.spearmanr(mask_loo, worker_mask)

        worker_scores[workers[i]] = c

    return worker_scores

def worker_scores_per_doc(corpus, annotype, score_type, pruned_workers=set()):

    worker_scores = {}

    for docid in corpus.docs:
        doc = corpus.docs[docid]

        if score_type in ['corr']:
            worker_scores_doc = worker_scores_doc_corr(doc, annotype, pruned_workers)
        elif score_type in ['prec', 'recl']:
            # TODO
            pass

        for wid in worker_scores_doc:
            if wid in worker_scores:
                worker_scores[wid][docid] = worker_scores_doc[wid]
            else:
                worker_scores[wid] = {docid: worker_scores_doc[wid]}

    for wid in worker_scores:
        print wid, np.mean(worker_scores[wid].values())


def worker_scores(corpus, annotype):
    pruned_workers = utils.get_pruned_workers(corpus, annotype)
    worker_scores_per_doc(corpus, annotype, 'corr', pruned_workers)

if __name__ == '__main__':
    anno_path = '../annotations/'

    #anno_fn = anno_path + 'PICO-annos-crowdsourcing.json'
    anno_fn = anno_path + 'PICO-annos-professional.json'

    # Loading corpus
    corpus = Corpus(doc_path = DOC_PATH)
    corpus.load_annotations(anno_fn)

    worker_scores(corpus, 'Participants')


