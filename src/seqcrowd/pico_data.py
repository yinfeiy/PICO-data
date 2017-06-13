import numpy as np
import sys
import util
from collections import defaultdict

sys.path.insert(0, '/mnt/data/workspace/nlp/PICO-data/src/')
from pico.corpus import Corpus
from pico import utils

ANNO_PATH = '/mnt/data/workspace/nlp/PICO-data/annotations/PICO-annos-crowdsourcing.json'
DOC_PATH = '/mnt/data/workspace/nlp/PICO-data/docs/'

def make_input(docs):

    all_input = []
    for docid in docs:
        spacydoc = docs[docid].spacydoc
        for sent in spacydoc.sents:
            for token in sent:
                t = u'{0} {1}_{2}_{3}'.format(\
                        token.text, docid, token.idx, token.idx+len(token) )
                all_input.append(t)
            all_input.append('')

    return all_input

def make_data(corpus, annotype):
    # All worder ids
    list_wid = []

    # Dictionary of worker ids  ## WHY is NEEDED ? ##
    dic_wid = {}

    # data (worker_id, doc_id, spans)
    data = []

    # Dictionary of doc ids
    dic_did_data = defaultdict(list)

    set_wid = set()
    data_size = 0
    for docid in corpus.docs:
        markups_offset = corpus.docs[docid].markups_offset.get(annotype, None)

        if not markups_offset:
            continue

        for wid, spans in markups_offset.items():
            set_wid.add(wid)
            spans = [ [s[0], s[1]-1] for s in spans ]  # The PICO format is (start, end), while HMMCrowd model takes (start, thru)
            data.append( (wid, docid, spans) )
            data_size += 1
            dic_did_data[docid].append(data_size-1)


    list_wid = sorted(list(set_wid))
    dic_wid = {wid:i for i, wid in enumerate(list_wid)}

    return (list_wid, dic_wid, data, dic_did_data)

def make_index(corpus, annotype):
    all_input = make_input(corpus.docs)
    features, labels = util.build_index(all_input)

    return (features, labels)


def is_inside(inv_labels, label, spans):
        if label not in inv_labels: return False
        label = inv_labels[label]
        pid, l, r = map(int, label.split('_'))
        for x, y in spans:
            if r >= x and l <= y: return True
        return False


def make_crowd_data(corpus, data, list_wid, dic_wid, dic_pid_data, features, labels):
    """
    make util.crowd_data
    """
    #make sentences and crowdlabs

    inv_labels = {v:k for (k,v) in labels.items()}

    sentences = []
    clabs = []

    for pid in corpus.docs:
        inp = make_input({pid: corpus.docs[pid]})

        sens = util.extract(inp, features, labels)
        if pid not in dic_pid_data:
            # TODO(yinfeiy): find out why it is missing here
            print pid,
            continue
        sentences.extend(sens)

        for sen in sens: # a sentence
            sen_clab = []
            for i in dic_pid_data[pid]: # a crowd annotation
                d = data[i]             # d = (wid, pid, spans)
                wlabs = [0] * len(sen)
                for j in range(len(sen)): # a word
                    if is_inside( inv_labels, sen[j].label, d[2]):
                        wlabs[j] = 1
                sen_clab.append(util.crowdlab(dic_wid[d[0]], int(d[1]), wlabs))

            clabs.append(sen_clab)

    return util.crowd_data(sentences, clabs)


def main(annotype, docids=None, max_num_worker=None, high_quality=False):

    if high_quality:
        print "[INFO] high_quality mode enabled, calculating low quality workers"
        corpus = Corpus(doc_path = DOC_PATH, verbose=False)
        corpus.load_annotations(ANNO_PATH, docids=docids)
        pruned_workers = {annotype: utils.get_pruned_workers(corpus, annotype)}
        print "[INFO] {0} workers are pruned because of low quality.".format(len(pruned_workers[annotype]))
    else:
        pruned_workers = {}

    corpus = Corpus(doc_path = DOC_PATH)
    corpus.load_annotations(ANNO_PATH, docids, max_num_worker=max_num_worker, pruned_workers=pruned_workers)

    list_wid, dic_wid, data, dic_did_data = make_data(corpus, annotype)
    features, labels = make_index(corpus, annotype)

    cd = make_crowd_data(corpus, data, list_wid, dic_wid, dic_did_data, features, labels)

    return (cd, list_wid, features, labels)


if __name__ == '__main__':
    annotype = 'Outcome'
    main(annotype, high_quality=True)

    # example docids = ['23549581']
    #main(annotype, docids=['23549581'], high_quality=True)

