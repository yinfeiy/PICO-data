import numpy as np
import scipy.stats as stats

def worker_scores_doc(doc, annotype, pruned_workers=set()):

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
            print "Warning: Only one worker for doc ", doc.docid
            c = 0.2
        elif len(worker_mask) == sum(worker_mask):
            c = 0
        else:
            mask_loo = (markup_mask - worker_mask) / (nworker-1)
            c, p = stats.spearmanr(mask_loo, worker_mask)

        worker_scores[workers[i]] = c

    return worker_scores


def get_pruned_workers(corpus, annotype):
    pruned_workers = set()

    worker_scores = {}
    for docid, doc in corpus.docs.items():
        ws_doc= worker_scores_doc(doc, annotype)
        for wid in ws_doc:
            if wid in worker_scores:
                worker_scores[wid][docid] = ws_doc[wid]
            else:
                worker_scores[wid] = {docid: ws_doc[wid]}

    for wid in worker_scores:
        ws = np.mean( worker_scores[wid].values() )
        if ws < 0.2:
            pruned_workers.add(wid)

    return pruned_workers


if __name__ == '__main__':
    from corpus import Corpus, Doc

    anno_path = '../../annotations/'
    anno_fn = anno_path + 'PICO-annos-crowdsourcing.json'
    gt_fn = anno_path + 'PICO-annos-professional.json'

    corpus = Corpus(doc_path = '../../docs/')
    corpus.load_annotations(anno_fn)
    corpus.load_groudtruth(gt_fn)

    pruned_workers = get_pruned_workers(corpus, annotype ='Participants')
    print pruned_workers
