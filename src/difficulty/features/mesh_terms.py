import json
import os
import math
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

MESH_DOC_DIR="/mnt/data/workspace/nlp/PICO-data/docs/mesh/mesh_tags/"
MESH_TREE_FILE="/mnt/data/workspace/nlp/PICO-data/docs/mesh/mtrees2018.bin"
#MESH_TERM_FEATURE_DICT="/mnt/data/workspace/nlp/PICO-data/src/difficulty/features/mesh_terms/terms_with_parents.dict"
MESH_TERM_FEATURE_DICT="/mnt/data/workspace/nlp/PICO-data/src/difficulty/features/mesh_terms/terms.dict"

def _load_mesh_tree():
    mesh_dict = {}
    tree_dict = {}
    with open(MESH_TREE_FILE) as fin:
        for line in fin:
            key, val= line.strip().split(";")
            mesh_dict[key] = val
            tree_dict[val] = key
    return mesh_dict, tree_dict

def _load_feature_dict(min_freq=5):
    feature_terms = {}
    with open(MESH_TERM_FEATURE_DICT) as fin:
        cnt = 0
        for line in fin:
            ts = line.strip().split("\t")
            term, freq = ts
            freq = int(freq)
            if freq < min_freq:
                continue
            feature_terms[term] = cnt
            cnt += 1
    return feature_terms


MESH_DICT, TREE_DICT = _load_mesh_tree()
MESH_TERMS = _load_feature_dict()

def get_feature_terms():
    feature_terms = MESH_TERMS.keys()
    feature_terms.sort(key=lambda x:MESH_TERMS[x])

    return feature_terms

def transform(mesh_terms_arr):
    vecs = []
    for mesh_terms in mesh_terms_arr:
        vec = np.zeros(len(MESH_TERMS))
        for term in mesh_terms:
            if term in MESH_TERMS:
                idx = MESH_TERMS[term]
                vec[idx] = 1
        vecs.append(vec)
    vecs = np.array(vecs)
    return vecs


def transform_by_docids(docids):
    doc_mesh_terms = get_mesh_terms(docids)
    return transform(doc_mesh_terms)


def get_ancestor_terms(mesh_terms, combine=True):
    result = []
    for mesh_term in mesh_terms:
        if mesh_term in MESH_DICT:
            mesh_id = MESH_DICT[mesh_term]
        else:
            result.append(set([mesh_term]))
            continue
        ancestor_ids = mesh_id.split('.')
        ancestors = set()
        for i in range(len(ancestor_ids)):
            id = ".".join(ancestor_ids[:i+1])
            term = TREE_DICT[id]
            ancestors.add(term)
        result.append(ancestors)

    if combine:
        tmp = set()
        for s in result:
            tmp = tmp.union(s)
        result = tmp
    return result


def get_mesh_terms(docids):
    doc_mesh_terms = []
    cnt = 0
    for did in docids:
        terms = set()
        mesh_doc_file = os.path.join(MESH_DOC_DIR, did+".txt")
        if not os.path.exists(mesh_doc_file):
            print "missing mesh terms for ", did
            continue

        with open(mesh_doc_file) as fin:
            for line in fin:
                term = line.strip()
                if term.find("/") > 0:
                    term = term.split("/")[0]
                terms.add(term)
        doc_mesh_terms.append(terms)
    return doc_mesh_terms


def histograms(doc_mesh_terms):
    count = defaultdict(int)
    for mesh_terms in doc_mesh_terms:
        for term in mesh_terms:
            count[term] += 1
    terms = count.keys()
    terms.sort(key=lambda x: count[x], reverse=True)
    with open("mesh_terms/terms.dict", "w+") as fout:
        for term in terms:
            cnt = count[term]
            fout.write("{0}\t{1}\n".format(term, cnt))
    vs = [math.log(v, 2) for v in count.values()]
    vs = [v for v in vs if v > 2 ]
    plt.hist(vs, bins=50)
    plt.savefig("mesh_terms/hist_terms.png")

    all_mesh_terms = []
    for mesh_terms in doc_mesh_terms:
        new_terms = get_ancestor_terms(mesh_terms)
        all_mesh_terms.append(new_terms)

    count = defaultdict(int)
    for mesh_terms in all_mesh_terms:
        for term in mesh_terms:
            count[term] += 1
    terms = count.keys()
    terms.sort(key=lambda x: count[x], reverse=True)
    with open("mesh_terms/terms_with_parents.dict", "w+") as fout:
        for term in terms:
            cnt = count[term]
            fout.write("{0}\t{1}\n".format(term, cnt))

    plt.clf()
    vs = [math.log(v, 2) for v in count.values()]
    vs = [v for v in vs if v > 2 ]
    plt.hist(vs, bins=50)
    plt.savefig("mesh_terms/hist_terms_with_parents.png")

if __name__ == "__main__":
    ifn = "/mnt/data/workspace/nlp/PICO-data/src/tmp/dataset/Intervention_abstract_train.data"
    docids = []
    with open(ifn) as fin:
        for line in fin:
            item = json.loads(line.strip())
            docids.append(item["sent_id"])
    mesh_terms = get_mesh_terms(docids)
    histograms(mesh_terms)
