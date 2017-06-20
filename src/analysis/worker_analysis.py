from pico.corpus import Corpus, Doc
from pico import utils

import numpy as np
import scipy.stats as stats
import metrics
import matplotlib.pyplot as plt
import random
from collections import defaultdict, Counter

random.seed(10)
DEFAULT_MAX_WORKERS = 1000
SCORETYPES = ['corr', 'prec', 'recl']

def worker_scores_doc_corr(doc, annotype, pruned_workers, max_workers=DEFAULT_MAX_WORKERS):
    # Leave One Out
    markups = doc.markups[annotype]
    workers = [w for w in markups.keys() if w not in pruned_workers]
    nworkers = len(workers)

    if nworkers > max_workers:
        random.shuffle(workers)
        workers = workers[:max_workers]
        workers.sort()
        nworkers = max_workers

    markup_mask = np.zeros(doc.ntokens)
    for i in range(nworkers):
        spans = markups[workers[i]]
        for span in spans:
            markup_mask[span[0]:span[1]] = markup_mask[span[0]:span[1]] + [1] * (span[1]-span[0])

    worker_scores = {}
    for i in range(nworkers):
        worker_mask = np.zeros(doc.ntokens)
        spans = markups[workers[i]]
        for span in spans:
            worker_mask[span[0]:span[1]] = [1] * (span[1]-span[0])

        if nworkers == 1:
            print "[Warn] Only one worker for doc {0}, do not calculate worker score.".format(doc.docid)
            continue

        elif len(worker_mask) == sum(worker_mask):
            c = 0
        else:
            mask_loo = (markup_mask - worker_mask) / (nworkers-1)
            c, p = stats.spearmanr(mask_loo, worker_mask)

        worker_scores[workers[i]] = c

    return worker_scores


def worker_scores_doc_helper(doc, annotype, scoretype, pruned_workers, max_workers=DEFAULT_MAX_WORKERS):
    markups = doc.markups[annotype]
    workers = [w for w in markups.keys() if w not in pruned_workers]
    nworkers = len(workers)

    if nworkers > max_workers:
        random.shuffle(workers)
        workers = workers[:max_workers]
        workers.sort()
        nworkers = max_workers

    worker_scores = {}
    if nworkers <= 1:
        print "[Warn] Only one worker for doc {0}, do not calculate worker score.".format(doc.docid)
    else:
        for wid in workers:
            worker_scores[wid] = []
        for i in range(nworkers-1):
            w1_spans = markups[workers[i]]
            w1_scores = []
            for j in range(i+1, nworkers):
                if i == j:
                    continue

                w2_spans = markups[workers[j]]
                score = metrics.metrics(w1_spans, w2_spans, doc.ntokens, scoretype)

                worker_scores[workers[i]].append(score)
                worker_scores[workers[j]].append(score)

    for wid in worker_scores:
        worker_scores[wid] = np.mean(worker_scores[wid])

    return worker_scores


def worker_scores_per_doc(docs, annotype, scoretype, pruned_workers=set(), max_workers=DEFAULT_MAX_WORKERS):

    worker_scores = {}

    for docid in docs:
        doc = docs[docid]

        if scoretype == 'corr':
            worker_scores_doc = worker_scores_doc_corr(doc, annotype, pruned_workers, max_workers)
        elif scoretype in ['prec', 'recl']:
            worker_scores_doc = worker_scores_doc_helper(doc, annotype, scoretype, pruned_workers, max_workers)

        for wid in worker_scores_doc:
            if wid in worker_scores:
                worker_scores[wid][docid] = worker_scores_doc[wid]
            else:
                worker_scores[wid] = {docid: worker_scores_doc[wid]}

    return worker_scores


def worker_hist_per_doc(worker_scores):
    doc_hist = defaultdict(int)
    for wid, scores in worker_scores.items():
        for docid in scores.keys():
            doc_hist[docid] += 1
    return doc_hist


def calculate_worker_scores(corpus, annotype, scoretype, max_workers=DEFAULT_MAX_WORKERS):
    pruned_workers = utils.get_pruned_workers(corpus, annotype)

    worker_scores = worker_scores_per_doc(corpus.docs, annotype, scoretype, pruned_workers, max_workers)
    return worker_scores


def plot_worker_hist(doc_hist, title, savefig=False):
    ct = Counter(doc_hist.values())
    plt.bar(ct.keys(), ct.values(), align='center', alpha=0.5)
    plt.title(title, fontsize=26)
    plt.xlabel("number of valid workers", fontsize=20)
    plt.ylabel("number of abstracts", fontsize=20)
    plt.xlim([0,8])
    if savefig:
        plt.savefig('hist_worker_{0}.png'.format(title.lower()))
    else:
        plt.show()

def plot_worker_scores_hist(worker_scores, title, savefig=False, gt=False):
    values = worker_scores.values()
    plt.clf()
    plt.hist(values, bins=20, alpha=0.5, edgecolor='w')
    plt.title(title, fontsize=26)
    plt.xlabel("worker scores", fontsize=20)
    plt.ylabel("number of workers", fontsize=20)
    plt.xlim([-0.2,1])
    if savefig:
        if gt:
            plt.savefig('hist_worker_scores_gt_{0}.png'.format(title.lower()))
        else:
            plt.savefig('hist_worker_scores_{0}.png'.format(title.lower()))
    else:
        plt.show()

def plot_tasks_per_worker_hist(worker_hist, title,savefig=False):
    values = worker_hist.values()
    values = [min(v, 100) for v in values]
    plt.clf()
    plt.hist(values, bins=30, alpha=0.5, edgecolor='w')
    plt.title(title + '({0})'.format(len(values)), fontsize=26)
    plt.xlabel("number of tasks completed", fontsize=20)
    plt.ylabel("number of workers", fontsize=20)
    if savefig:
        plt.savefig('tasks_per_worker_{0}.png'.format(title.lower()))
    else:
        plt.show()

def output_worker_scores(worker_scores, valid_docs, output_filename):
    fout = open(output_filename, 'w+')
    for wid in worker_scores:
        ostr = '{0},'.format(wid)

        scores = worker_scores[wid]
        valid_scores = [scores[docid] for docid in scores if docid in valid_docs]
        fout.write('{0}, {1}, {2:.3f}\n'.format(wid, len(valid_scores), np.mean(valid_scores)))
    fout.close()


def count_tasks_per_worker(corpus, plot=False):
    worker_count_all = defaultdict(dict)
    for annotype in utils.ANNOTYPES:
        worker_count = defaultdict(int)
        for docid in corpus.docs:
            annos = corpus.get_doc_annos(docid, annotype)
            if not annos:
                continue
            for wid in annos:
                worker_count[wid] += 1
        plot_tasks_per_worker_hist(worker_count, annotype, savefig=True)
        for wid in worker_count:
            worker_count_all[wid][annotype] = worker_count[wid]

    # How many of them come back for other annotypes
    vals = [len(worker_count_all[wid].keys()) for wid in worker_count_all]
    print Counter(vals)

def main(corpus, plot=False):
    #count_tasks_per_worker(corpus, plot=plot)

    # Worker Scores
    for annotype in utils.ANNOTYPES:
        worker_scores_tmp = calculate_worker_scores(corpus, annotype, 'corr', DEFAULT_MAX_WORKERS)

        # number of workers per doc
        doc_hist = worker_hist_per_doc(worker_scores_tmp)
        # plot_worker_hist(doc_hist, annotype)

        mean_scores = dict([(wid, np.mean(worker_scores_tmp[wid].values())) for wid in worker_scores_tmp])
        print annotype, "number of wokers <0.2: ", sum([ 1 if s < 0.2 else 0 for s in mean_scores.values()])
        if plot:
            plot_worker_scores_hist(mean_scores, annotype, savefig=True)


if __name__ == '__main__':
    doc_path = '../docs/'

    anno_fn = '../annotations/PICO-annos-crowdsourcing.json'
    gt_fn = '../annotations/PICO-annos-professional.json'

    # Loading corpus
    corpus = Corpus(doc_path = doc_path)
    corpus.load_annotations(anno_fn)
    corpus.load_groundtruth(gt_fn)

    main(corpus, plot=True)

