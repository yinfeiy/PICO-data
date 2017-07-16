from pico.corpus import Corpus, Doc
from pico import utils
from analysis.worker_analysis import worker_scores_doc_corr, worker_scores_doc_helper
from analysis.worker_analysis import worker_scores_doc_corr_gt, worker_scores_doc_gt_helper
from analysis.worker_analysis import worker_scores_sent_corr
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import json

def doc_scorer(corpus):
    doc_scores = defaultdict(dict)

    for annotype in utils.ANNOTYPES:
        pruned_workers = utils.get_pruned_workers(corpus, annotype)

        for docid in corpus.docs:
            doc = corpus.docs[docid]

            for scoretype in ['corr', 'prec', 'recl']:
                if scoretype == 'corr':
                    worker_scores = worker_scores_doc_corr(doc, annotype, pruned_workers)
                    worker_scores_gt = worker_scores_doc_corr_gt(doc, annotype, pruned_workers)
                elif scoretype in ['prec', 'recl']:
                    worker_scores = worker_scores_doc_helper(doc, annotype, scoretype, pruned_workers)
                    worker_scores_gt = worker_scores_doc_gt_helper(doc, annotype, scoretype, pruned_workers)
                else:
                    worker_scores, worker_scores_gt = {}, {}

                doc_score = np.mean(worker_scores.values())
                if not np.isnan(doc_score):
                    doc_scores[docid][annotype+'_'+scoretype] = doc_score
                else:
                    doc_scores[docid][annotype+'_'+scoretype] = None

                doc_score_gt = np.mean(worker_scores_gt.values())
                if not np.isnan(doc_score_gt):
                    doc_scores[docid][annotype+'_'+scoretype+'_'+'gt'] = doc_score_gt
                else:
                    doc_scores[docid][annotype+'_'+scoretype+'_'+'gt'] = None

    return doc_scores

def get_span_text(spacydoc, spans):
    mask = np.zeros(len(spacydoc)+1) # append a non span at the end
    for span in spans:
        mask[span[0]:span[1]] += 1

    for sent in spacydoc.sents:
        if np.sum(mask[sent.start:sent.end]) > 0:
            #for idx in range(sent.start, sent.end):
            #    if mask[idx] >= 1:
            #        print '[ ' + spacydoc[idx].text + '_' + str(mask[idx]) + ' ]',
            #    else:
            #        print spacydoc[idx].text,
            #print ''
            mask[sent.start:sent.end] = 1
        else:
            #print "hello"
            pass

    # convert mask to final spans
    spans = []
    if mask[0] == 1:
        sidx = 0

    for idx, v in enumerate(mask[1:], 1):
        if v==1 and mask[idx-1] == 0: # start of span
            sidx = idx
        elif v==0 and mask[idx-1] == 1 : # end of span
            eidx = idx
            spans.append( (sidx, eidx) )

    text = ""
    for span in spans:
        span = spacydoc[span[0]:span[1]]
        for token in span:
            text += " " + token.text

        if not token.is_punct:
            text += " ."

    return text.strip()


def save_doc_scores(corpus, doc_scores, ofn=None):
    if not ofn:
        ofn = './difficulty.json'

    with open(ofn, 'w+') as fout:
        for docid in corpus.docs:
            doc_scores[docid]['text'] = corpus.get_doc_tokenized_text(docid)
            doc_scores[docid]['docid'] = docid

            #annos = corpus.get_doc_annos(docid)
            annos = corpus.get_doc_aggregation(docid)

            spacydoc = corpus.get_doc_spacydoc(docid)
            all_spans = []
            for annotype in annos.keys():
                if isinstance(annos[annotype], list):
                    spans = annos[annotype]
                elif isinstance(annos[annotype], dict):
                    spans = []
                    for ss in annos[annotype].values():
                        spans.extend(ss)
                all_spans.extend(spans)

                span_text = get_span_text(spacydoc, spans)
                doc_scores[docid]['{0}_text'.format(annotype)] = span_text
            #span_test = get_span_text(spacydoc, all_spans)
            #doc_scores[docid]['span_text'.format(annotype)] = span_text

            ostr = json.dumps(doc_scores[docid])
            fout.write(ostr + '\n')

    return ofn

def plot_score_dist(scores, scoretype='corr', savefig=False, figname=None):
    keys = ['min'];
    keys.extend(utils.ANNOTYPES)

    colors = dict(zip(keys, ['red', 'green', 'blue', 'magenta']))
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(16,10))
    axs = [ax1, ax2, ax3, ax4]
    axs[0].set_title('Histogram of scores', fontsize=26)
    for idx, annotype in enumerate(keys):
        scores_arr = []
        for id in scores:
            s = 10000
            if annotype == 'min':
                for annotype_iter in utils.ANNOTYPES:
                    tmp = scores[id][annotype_iter+'_'+scoretype]
                    s = min(s, tmp) if tmp else s
            else:
                tmp = scores[id][annotype+'_'+scoretype]
                s = tmp if tmp else s
            if s != 10000:
                scores_arr.append(s)
        mean = np.average(scores_arr)
        std = np.std(scores_arr)

        lines = np.linspace( min(scores_arr), max(scores_arr), 100 )
        nd = stats.norm.pdf(lines, mean, std)

        axs[idx].hist(scores_arr, 50, facecolor=colors[annotype], alpha=0.5, label=annotype, normed=True, edgecolor='w')

        axs[idx].plot(lines, nd, color='b', linewidth=3)
        axs[idx].grid(True)
        #axs[idx].set_ylabel('Count (normed)', fontsize=20)
        axs[idx].legend(['{0} Mean": {1:.3f}; Std: {2:.3f}'.format(annotype, mean, std)], fontsize=16, loc=2)
    plt.xlabel('Scores', fontsize=20)
    axs[1].set_ylabel('Count (normed)', fontsize=20)
    if savefig:
        plt.savefig(figname)
    else:
        plt.show()

def load_doc_scores(ifn, is_dict=False):
    doc_scores = []
    with open(ifn) as fin:
        for line in fin:
            doc_scores.append(json.loads(line.strip()))
    if is_dict:
        doc_scores = dict(zip([d['docid'] for d in doc_scores], doc_scores))
    return doc_scores

def inter_annotype_correlation(doc_scores, scoretype='corr'):
    if isinstance(doc_scores, dict):
        doc_scores = doc_scores.values()
    for annotype_1, annotype_2 in combinations(utils.ANNOTYPES, 2):
        ss1, ss2 = [], []
        for item in doc_scores:
            s1 = item[annotype_1+'_'+scoretype]
            s2 = item[annotype_2+'_'+scoretype]
            if s1 and s2:
                ss1.append(s1), ss2.append(s2)
        print "Inter annotype corrleation: ", annotype_1, annotype_2
        print stats.pearsonr(ss1, ss2)
        print stats.spearmanr(ss1, ss2)

def doc_score_anno_quality(doc_scores, scoretype='corr'):
    if isinstance(doc_scores, dict):
        doc_scores = doc_scores.values()
    for annotype in utils.ANNOTYPES:
        ss1, ss2 = [], []
        for item in doc_scores:
            s1 = item[annotype+'_'+scoretype]
            s2 = item[annotype+'_'+scoretype + '_' + 'gt']
            if s1 and s2:
                ss1.append(s1), ss2.append(s2)
        print "Annotation quality for : ", annotype,
        print stats.pearsonr(ss1, ss2)
        print stats.spearmanr(ss1, ss2)

# TODO(yinfeiy): either remove or finish the sentence scorer, currently focus on doc level
def sent_scorer(corpus):
    sent_scores = defaultdict(dict)

    for annotype in utils.ANNOTYPES:
        pruned_workers = utils.get_pruned_workers(corpus, annotype)

        for docid in corpus.docs:
            doc = corpus.docs[docid]

            for scoretype in ['corr', 'prec', 'recl']:
                if scoretype == 'corr':
                    worker_scores = worker_scores_sent_corr(doc, annotype, pruned_workers)

if __name__ == '__main__':
    doc_path = '../docs/'

    anno_fn = '../annotations/PICO-annos-crowdsourcing.json'
    agg_fn = '../annotations/PICO-annos-crowdsourcing-agg.json'
    gt_fn = '../annotations/PICO-annos-professional.json'
    agg_ids = 'HMMCrowd'

    ofn = './difficulty/difficulty_with_span_sents.json'

    # Loading corpus
    if True:
        corpus = Corpus(doc_path = doc_path)
        corpus.load_annotations(anno_fn)
        corpus.load_groundtruth(gt_fn)
        corpus.load_aggregation(agg_fn, agg_ids)

        doc_scores = doc_scorer(corpus)
        save_doc_scores(corpus, doc_scores, ofn)
    else:
        doc_scores = load_doc_scores(ofn, is_dict=True)
    inter_annotype_correlation(doc_scores)
    doc_score_anno_quality(doc_scores, scoretype='corr')
    #plot_score_dist(doc_scores, savefig=False, figname='./hist_sent_scores.png')
