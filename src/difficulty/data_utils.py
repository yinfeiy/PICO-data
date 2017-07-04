import json, os
import re
import random
import numpy as np

random.seed(10)

CPATH = os.path.dirname(os.path.realpath(__file__))
DATASET =  os.path.join(CPATH, 'difficulty.json')

ANNOTYPES = ['Participants', 'Intervention', 'Outcome']
SCORETYPES = ['corr', 'prec', 'recl']
DEFAULT_ANNOTYPE = 'min'
DEFAULT_SCORETYPE = 'corr'

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def calculate_percentiles(docs, field='score', new_field='percentile'):
    scores = [doc[field] for doc in docs]
    num = len(scores)
    idxs = range(num)
    idxs.sort(key=lambda x:scores[x], reverse=False)
    for rank, idx in enumerate(idxs, 1):
        docs[idx][new_field] = round(rank*100.0/num, 0)

    return docs


def split_train_test(docs, development_set=0):
    gt_keys = []
    for annotype in ANNOTYPES:
        for scoretype in SCORETYPES:
            key = '_'.join([annotype, scoretype, 'gt'])
            gt_keys.append(key)

    test_docids = set()
    for doc in docs:
        for key in gt_keys:
            if doc.get(key, None):
                test_docids.add(doc['docid'])
                break

    train_docids = [ doc['docid'] for doc in docs if doc['docid'] not in test_docids ]
    if development_set:
        random.shuffle(train_docids)
        th = int(len(train_docids)*(1-development_set))
        dev_docids = set(train_docids[th:])
        train_docids = set(train_docids[:th])
    else:
        train_docids = set(train_docids)
        dev_docids = set()

    return train_docids, dev_docids, test_docids


def extract_text(docs, percentile=True, gt=False):
    text = []
    ys = []
    for doc in docs:
        if gt and 'gt' not in doc:
            continue

        text.append(clean_str(doc['text']))
        if gt:
            if percentile:
                ys.append(doc['percentile_gt'])
            else:
                ys.append(doc['gt'])
        else:
            if percentile:
                ys.append(doc['percentile'])
            else:
                ys.append(doc['score'])

    return text, ys


def load_dataset(development_set=0.2, annotype=DEFAULT_ANNOTYPE, scoretype=DEFAULT_SCORETYPE):
    docs = []
    docs_raw = []
    with open(DATASET) as fin:
        for line in fin:
            doc = {}
            item = json.loads(line.strip())
            docs_raw.append(item)
            if annotype == 'min':
                scores, gts = [], []
                for at in ANNOTYPES:
                    s = item[at+'_'+scoretype]
                    g = item[at+'_'+scoretype+'_'+'gt']
                    if s: scores.append(s)
                    if g: gts.append(g)
                doc['score'] = np.min(scores) if scores else None
                doc['gt'] = np.min(gts) if gts else None
            elif annotype in ANNOTYPES:
                doc['score'] = item[annotype+'_'+scoretype]
                doc['gt'] = item[annotype+'_'+scoretype+'_'+'gt']
            else:
                raise 'To be implementated'

            doc['text'] = item['text']
            doc['docid'] = item['docid']
            docs.append(doc)

    docs = calculate_percentiles(docs)

    train_docids, dev_docids, test_docids = split_train_test(
            docs_raw, development_set=development_set)

    train_docs = [doc for doc in docs if doc['docid'] in train_docids and doc['score']]
    dev_docs   = [doc for doc in docs if doc['docid'] in dev_docids and doc['score']]
    test_docs  = [doc for doc in docs if doc['docid'] in test_docids and doc['gt'] and doc['score']]

    test_docs = calculate_percentiles(test_docs, field='gt', new_field='percentile_gt')

    train_text, y_train = extract_text(train_docs)
    dev_text, y_dev = extract_text(dev_docs)
    test_text, y_test = extract_text(test_docs, gt=True)

    if development_set:
        return train_text, y_train, dev_text, y_dev, test_text, y_test
    else:
        return train_text, y_train, test_text, y_test


if __name__ == '__main__':
    items = load_dataset()
    for item in items:
        print len(item)