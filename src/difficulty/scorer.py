from pico.corpus import Corpus, Doc
from pico import utils
from analysis.worker_analysis import worker_scores_doc_corr, worker_scores_doc_helper
from collections import defaultdict

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
                elif scoretype in ['perc', 'recl']:
                    worker_scores = worker_scores_doc_helper(doc, annotype, scoretype, pruned_workers)

                doc_score = np.mean(worker_scores.values())
                if not np.isnan(doc_score):
                    doc_scores[docid][annotype+'_'+scoretype] = doc_score
                else:
                    doc_scores[docid][annotype+'_'+scoretype] = None


    return doc_scores

def save_doc_scores(corpus, doc_scores, ofn=None):
    if not ofn:
        ofn = './.difficulty.json'

    with open(ofn, 'w+') as fout:
        for docid in corpus.docs:
            doc_scores[docid]['text'] = corpus.get_doc_tokenized_text(docid)
            doc_scores[docid]['docid'] = docid
            ostr = json.dumps(doc_scores[docid])
            fout.write(ostr + '\n')

    return ofn

def plot_doc_score_dist(doc_scores, scoretype='corr', savefig=False):
    keys = ['min'];
    keys.extend(utils.ANNOTYPES)

    colors = dict(zip(keys, ['red', 'green', 'blue', 'magenta']))
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(16,10))
    axs = [ax1, ax2, ax3, ax4]
    axs[0].set_title('Histogram of doc scores', fontsize=26)
    for idx, annotype in enumerate(keys):
        scores = []
        for docid in doc_scores:
            s = 10000
            if annotype == 'min':
                for annotype_iter in utils.ANNOTYPES:
                    tmp = doc_scores[docid][annotype_iter+'_'+scoretype]
                    s = min(s, tmp) if tmp else s
            else:
                tmp = doc_scores[docid][annotype+'_'+scoretype]
                s = tmp if tmp else s
            if s != 10000:
                scores.append(s)
        mean = np.average(scores)
        std = np.std(scores)

        lines = np.linspace( min(scores), max(scores), 100 )
        nd = stats.norm.pdf(lines, mean, std)

        axs[idx].hist(scores, 50, facecolor=colors[annotype], alpha=0.5, label=annotype, normed=True, edgecolor='w')

        axs[idx].plot(lines, nd, color='b', linewidth=3)
        axs[idx].grid(True)
        #axs[idx].set_ylabel('Count (normed)', fontsize=20)
        axs[idx].legend(['{0} Mean": {1:.3f}; Std: {2:.3f}'.format(annotype, mean, std)], fontsize=16, loc=2)
    plt.xlabel('Scores', fontsize=20)
    axs[1].set_ylabel('Count (normed)', fontsize=20)
    if savefig:
        plt.savefig('hist_doc_scores.png')
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

if __name__ == '__main__':
    doc_path = '../docs/'

    anno_fn = '../annotations/PICO-annos-crowdsourcing.json'
    gt_fn = '../annotations/PICO-annos-professional.json'

    ofn = './difficulty/difficulty.json'

    # Loading corpus
    if False:
        corpus = Corpus(doc_path = doc_path)
        corpus.load_annotations(anno_fn)
        corpus.load_groundtruth(gt_fn)

        doc_scores = doc_scorer(corpus)
        save_doc_scores(corpus, doc_scores, ofn)
    else:
        doc_scores = load_doc_scores(ofn, is_dict=True)
    plot_doc_score_dist(doc_scores, savefig=True)
