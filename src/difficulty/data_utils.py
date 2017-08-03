import json, os
import re
import random
import numpy as np

random.seed(10)

CPATH = os.path.dirname(os.path.realpath(__file__))
DATASET =  os.path.join(CPATH, 'difficulty_with_span_sents.json')
DATASET_PROB =  os.path.join(CPATH, 'difficulty_annotated.json')

ANNOTYPES = ['Participants', 'Intervention', 'Outcome']
SCORETYPES = ['corr', 'prec', 'recl']
DEFAULT_ANNOTYPE = 'Participants'
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
    if isinstance(scores[0], list):
        scores = np.array(scores)
        scores[1,1] = np.nan
        ranks = scores.argsort(axis=0).argsort(axis=0).astype(np.float32) + 1
        idxs = np.argwhere(np.isnan(scores))
        for idx in idxs:
            ranks[idx[0], idx[1]] = np.nan
        pctls = np.round(ranks/np.nanmax(ranks, 0)*100, 0)/100
        for idx, doc in enumerate(docs):
            doc[new_field] = pctls[idx,:]
    else:
        num = len(scores)
        idxs = range(num)
        idxs.sort(key=lambda x:scores[x], reverse=False)
        for rank, idx in enumerate(idxs, 1):
            docs[idx][new_field] = round(rank*100.0/num, 0) / 100.0

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

def extract_pos(docs, docids=None):
    pos = []
    for doc in docs:
        if docids and doc['docid'] not in docids:
            continue

        pos.append(doc['pos'])
    return pos

def extract_text(docs, percentile=True, gt=False):
    text = []
    ys = []
    docids = []
    for doc in docs:
        if gt and 'gt' not in doc:
            continue

        docids.append(doc['docid'])
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

    return docids, text, ys


def imputation(data):
    data = np.array(data)
    if len(data.shape) == 1:
        data = np.expand_dims(data, 1)

    cols = data.shape[1]
    default_score = np.nanmean(data, 0)
    for col in range(cols):
        idxs = np.argwhere(np.isnan(data[:,col]))
        data[idxs, col] = default_score[col]

    return data

def load_dataset_prob(development_set=0.2, annotype=DEFAULT_ANNOTYPE, scoretype=DEFAULT_SCORETYPE):
    max_sents=100
    docs = []
    docs_raw = []
    with open(DATASET_PROB) as fin:
        for line in fin:
            doc = {}
            item = json.loads(line)
            docs_raw.append(item)
            parsed_text = item['parsed_text']

            if annotype == 'multitask':
                # (TODO)
                pass
            elif annotype in ANNOTYPES:
                doc['score'] = item[annotype+'_'+scoretype]
                doc['gt'] = item[annotype+'_'+scoretype+'_'+'gt']

                sents = parsed_text['sents']
                sent_scores = [sent['{0}_prob'.format(annotype)] for sent in sents]
                sorted_idx = np.argsort(sent_scores)[::-1]

                text = ''
                pos = ''
                for idx in sorted_idx[:max_sents]:
                    if idx == 0 or sent_scores[idx] >= 0:
                        sent_text = ' '.join([token[0] for token in sents[idx]['tokens']])
                        sent_pos = ' '.join([token[2] for token in sents[idx]['tokens']])
                        text += sent_text.strip() + ' '
                        pos += sent_pos.strip() + ' '
                doc['text'] = text
                doc['pos'] = pos
                #if doc['gt'] != None:
                #    print item['docid']
                #    print text
                #    exit()
            else:
                raise 'To be implementated'

            doc['docid'] = item['docid']
            docs.append(doc)

    docs = calculate_percentiles(docs)

    train_docids, dev_docids, test_docids = split_train_test(
            docs_raw, development_set=development_set)

    train_docs = [doc for doc in docs if doc['docid'] in train_docids and doc['score']]
    dev_docs   = [doc for doc in docs if doc['docid'] in dev_docids and doc['score']]
    test_docs  = [doc for doc in docs if doc['docid'] in test_docids and doc['gt'] and doc['score']]

    test_docs = calculate_percentiles(test_docs, field='gt', new_field='percentile_gt')

    train_docids, train_text, y_train = extract_text(train_docs)
    dev_docids, dev_text, y_dev = extract_text(dev_docs)
    test_docids, test_text, y_test = extract_text(test_docs, gt=True)

    train_pos = extract_pos(train_docs, train_docids)
    dev_pos = extract_pos(dev_docs, dev_docids)
    test_pos = extract_pos(test_docs, test_docids)

    if development_set:
        return train_text, train_pos, y_train, dev_text, dev_pos, y_dev, test_text, test_pos, y_test
    else:
        return train_text, train_pos, y_train, test_text,test_pos,  y_test

def load_dataset(development_set=0.2, annotype=DEFAULT_ANNOTYPE, scoretype=DEFAULT_SCORETYPE, span_text=False):
    docs = []
    docs_raw = []
    with open(DATASET) as fin:
        for line in fin:
            doc = {}
            item = json.loads(line)
            docs_raw.append(item)
            if annotype == 'multitask':
                scores, gts = [], []
                for at in ANNOTYPES:
                    s = item[at+'_'+scoretype]
                    g = item[at+'_'+scoretype+'_'+'gt']
                    if s: scores.append(s)
                    else: scores.append(np.nan)
                    if g: gts.append(g)
                    else: gts.append(np.nan)
                doc['score'] = scores
                doc['gt'] = gts
            elif annotype in ANNOTYPES:
                doc['score'] = item[annotype+'_'+scoretype]
                doc['gt'] = item[annotype+'_'+scoretype+'_'+'gt']
            else:
                raise 'To be implementated'

            if span_text:
                key = 'span_text' if annotype == 'multitask' else '{0}_text'.format(annotype)
                #key='span_text'
                doc['text'] = item.get(key, item['text'])
            else:
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

    train_docids, train_text, y_train = extract_text(train_docs)
    dev_docids, dev_text, y_dev = extract_text(dev_docs)
    test_docids, test_text, y_test = extract_text(test_docs, gt=True)

    if development_set:
        return train_text, y_train, dev_text, y_dev, test_text, y_test
    else:
        return train_text, y_train, test_text, y_test


if __name__ == '__main__':
    a = np.array([[1,2,3],[1,np.nan, np.nan], [3,4.1,np.nan]])
    a = imputation(a)
    print a
    #items = load_dataset()
    #for item in items:
    #    print len(item)
