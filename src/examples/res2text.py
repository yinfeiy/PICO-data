import sys
sys.path.insert(0, '/mnt/data/workspace/nlp/PICO-data/src/')

from pico.corpus import Corpus, Doc
from pico import utils
import json

if __name__ == '__main__':

    doc_path = '../../docs/'

    anno_fn = '/mnt/data/workspace/nlp/PICO-data/results_to_evaluate/PICO-annos-dw.json'

    gt_fn = '../../annotations/PICO-annos-professional.json'
    gt_wids = None

    docids = utils.docs_with_gt(gt_fn)

    # Loading corpus
    corpus = Corpus(doc_path = doc_path)
    corpus.load_annotations(anno_fn, docids)
    corpus.load_groundtruth(gt_fn, gt_wids) # It will load all annotators if wid is None

    annotypes = ['Outcome']
    for annotype in annotypes:
        for docid in corpus.docs:
            corpus.get_doc_annos(docid, annotype, text=True)
            exit()
