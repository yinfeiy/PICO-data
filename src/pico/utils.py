import numpy as np
import scipy.stats as stats
import json

ANNOTYPES = ['Participants', 'Intervention', 'Outcome']

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
            # print "[Warn] Only one worker for doc ", doc.docid
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


def get_pruned_workers_cache(annotype):
    if annotype == "Participants":
        worker_ids="A1APSNKUBXUCHY, A32JDVCVBNX67D, A3N7Q68T1JYMH3, A1SNDKYG0M49MS, A251UAE2STZOW3, A1IM8G6UNAGQ7S, A2LI31K2R4ER3C, A26BHQZCY7GRNP, A2341KCW7BI2NS, A382SL9ROIY1P6, A3FBTN97OT60R1, A23FUSSP74TO6W, A2V2MRSY9CG6Q1, AF5GLPZ5D1CF7, AEPHQ3XYCQ0CG, A33D5LFU31S89H, A1CCC1JZ9WE6KQ, A2Q5AYCZ0H2X8L, A3K743OHIF6ZVZ"
    elif annotype == "Intervention":
        worker_ids="AQXVN6YASXT6B, A1W952WJZBCE6N, A24VQRAIVS44NS, AQ10ACQ5021ES, A1SNDKYG0M49MS, A8H1FRM2MGN3V, A1IM8G6UNAGQ7S, A2OODTTTXZQBK9, A13FVM2C914A3H, A1VUTSS76IXXGC, A3AEGALX2Q1G2A, A2V2MRSY9CG6Q1, A209JC1S9D760N, A3M7MUXRH4CWHZ, A2VFN2WZ2A14IN, A10RBQRCM64EXW, A1UNPK6SMRTR71, A3LJTA6I12LE4Y, A1P6L6W6TA5NJ"
    elif annotype == "Outcome":
        worker_ids="A1LQKMSG4GT8N, ATUR98N8W23Q3, AN8YSX80N13B2, A1WDU40K0BZBAU, A7HN9HZWPIYIC, A1ZQRFTDDNETVI, AMTJPX8YLKTB5, ASRHTFLS612Y2, A263L8AUIUNB6Z, A1XNXRN8FRC4L6, A1C3FD08T4X50W, A2B5IMHREFHIE2, AMBIH4E0BRGW6, A1W9S81A82PM9M, A2ICL29CB31UG2, A2HNNNCB7WM8VD, A3TKUXUTDX6FBF, A3PF7ISCU865Y6, A1112Z9TKSL75K, A2VT0HKHSGWTL, A1WWO6GOW6NQY9, A355LNQ2VDZ7FE, A1FR59CO9Q41FC, A2ZGQUSBB0TMK4, A2DVFA1B7NMN0F, A8BZ9J8RSJWFD, AA1WPIKNEY5IW, A2T0LDVQO159I2, A261AMC0EJ9MD8, A2C47N2IM5SYGY, A3OT61DL0DY283, A1KN8VSDTHR84J, A37ADSGZMTUHBI, A30U1CYNLWL531, A28FA9S6Y1M1I7, A3PDQ3DJGJA78Y"

    worker_ids = worker_ids.split(",")
    return worker_ids


def get_spans(mask):
    mask.append(0)  # append a non span

    spans = []
    if mask[0] == 1:
        sidx = 0

    for idx, v in enumerate(mask[1:], 1):
        if v==1 and mask[idx-1] == 0: # start of span
            sidx = idx
        elif v==0 and mask[idx-1] == 1 : # end of span
            eidx = idx
            spans.append( (sidx, eidx) )
    return spans


def get_reverse_spans(mask):
    mask.append(1)

    spans_reverse = []
    if mask[0] == 0:
        sidx = 0
    for idx, v in enumerate(mask[1:], 1):
        if v==0 and mask[idx-1] == 1: # start of span
            sidx = idx
        elif v==1 and mask[idx-1] == 0: # end of span
            eidx = idx
            spans_reverse.append( (sidx, eidx) )
    return spans_reverse


def span2mask(spans, num):
    mask = np.zeros(num)

    for span in spans:
        st, et = span
        if st > num: continue
        if et > num: et = num
        mask[st:et] = 1
    return mask


def docs_with_gt(gt_fn):
    docids = []
    with open(gt_fn) as fin:
        for line in fin:
            item = json.loads(line.strip())
            docids.append( item['docid'] )
    return docids


if __name__ == '__main__':
    from corpus import Corpus, Doc

    anno_path = '../../annotations/'
    anno_fn = anno_path + 'PICO-annos-crowdsourcing.json'
    gt_fn = anno_path + 'PICO-annos-professional.json'

    corpus = Corpus(doc_path = '../../docs/')
    corpus.load_annotations(anno_fn)
    corpus.load_groundtruth(gt_fn)

    pruned_workers = get_pruned_workers(corpus, annotype ='Outcome')
    print ", ".join(pruned_workers)
