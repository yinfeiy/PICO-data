from pico.corpus import Corpus, Doc
from pico import utils
from analysis.worker_analysis import worker_scores_doc_corr
from collections import defaultdict

import numpy as np

def main(corpus):
    doc_scores = defaultdict(dict)

    for annotype in utils.ANNOTYPES:
        pruned_workers = utils.get_pruned_workers(corpus, annotype)

        for docid in corpus.docs:
            doc = corpus.docs[docid]
            worker_scores = worker_scores_doc_corr(doc, annotype, pruned_workers)
            doc_score = np.mean(worker_scores.values())
            if not np.isnan(doc_score):
                doc_scores[docid][annotype] = doc_score

        break

if __name__ == '__main__':
    doc_path = '../docs/'

    anno_fn = '../annotations/PICO-annos-crowdsourcing.json'
    gt_fn = '../annotations/PICO-annos-professional.json'

    # Loading corpus
    corpus = Corpus(doc_path = doc_path)
    corpus.load_annotations(anno_fn)
    corpus.load_groundtruth(gt_fn)

    main(corpus)
