from pico.corpus import Corpus, Doc
from pico import utils

from collections import defaultdict
import numpy as np
import json

def mv_doc(doc, annotype, pruned_workers):
    markups = doc.markups[annotype]

    workers = [w for w in markups.keys() if w not in pruned_workers]

    mask = np.zeros(doc.ntokens)

    for wid in workers:
        spans = markups[wid]
        w_mask = utils.span2mask(spans, doc.ntokens)
        mask += w_mask

    th = len(workers) / 2.0

    label = mask >= th
    label = label.astype(int)

    mv_spans = utils.get_spans(label.tolist())

    new_spans = []
    for span in mv_spans:
        st = doc.spacydoc[span[0]]
        et = doc.spacydoc[span[1]-1]
        new_spans.append([st.idx, et.idx + len(et)])

    return new_spans


def mv_corpus(corpus, annotype):
    pruned_workers = utils.get_pruned_workers(corpus, annotype)

    results = {}

    for docid in corpus.docs:
        doc = corpus.docs[docid]

        mv_spans = mv_doc(doc, annotype, pruned_workers)
        results[docid] = mv_spans

    return results

def process(corpus, ofn, annotypes=utils.ANNOTYPES):

    mv_res = defaultdict(dict)
    for annotype in annotypes:
        mv_res_tmp = mv_corpus(corpus, annotype)

        for docid in mv_res_tmp:
            mv_res[docid][annotype] = dict(mv=mv_res_tmp[docid])

    docids = mv_res.keys()
    docids.sort()

    fout = open(ofn, 'w+')
    for docid in docids:
        item = mv_res[docid]
        item['docid'] = docid
        fout.write(json.dumps(item) + '\n')
    fout.close()


def docs_with_gt(gt_fn):
    docids = []
    with open(gt_fn) as fin:
        for line in fin:
            item = json.loads(line.strip())
            docids.append( item['docid'] )
    return docids

def filter_docs(annotype, min_workers):
    ANNO_PATH = '/mnt/data/workspace/nlp/PICO-data/annotations/PICO-annos-crowdsourcing.json'

    docids = []
    with open(ANNO_PATH) as fin:
        for line in fin:
            item = json.loads(line.strip())
            annos = item.get(annotype, {})
            nworkers = len(annos.keys())
            if nworkers >= min_workers:
                docids.append(item['docid'])
    return docids

if __name__ == '__main__':
    doc_path = '../docs/'

    anno_fn = '../annotations/PICO-annos-crowdsourcing.json'
    gt_fn = '../annotations/PICO-annos-professional.json'

    #docids = docs_with_gt(gt_fn)
    docids = None
    high_quality = True
    max_num_worker = 3

    pruned_workers = {}
    if high_quality:
        corpus_raw = Corpus(doc_path = doc_path, verbose=False)
        corpus_raw.load_annotations(anno_fn, docids=docids)
        for annotype in utils.ANNOTYPES:
            pruned_workers[annotype] = utils.get_pruned_workers(corpus_raw, annotype)

    print pruned_workers
    for annotype in utils.ANNOTYPES:
        #docids = filter_docs(annotype, min_workers=6)
        # Loading corpus
        corpus = Corpus(doc_path = doc_path)
        corpus.load_annotations(anno_fn, docids, max_num_worker=max_num_worker, pruned_workers=pruned_workers)

        process(corpus, ofn='./aggregated_results/{0}-aggregated-mv_max3.json'.format(annotype), annotypes=[annotype])
