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
        scores = np.array(scores, dtype=np.float32)
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
    if not docids:
        docids = docs.keys()

    for docid in docids:
        pos.append(docs[docid]['pos'])
    return pos

def extract_text(docs, gt=False, percentile=True):
    text = []
    ys = []
    for doc in docs:
        text.append(clean_str(doc['text']))
        if gt:
            if 'gt' not in doc:
                if isinstance(doc['score'], list):
                    ys.append([np.nan]*len(doc['score']))
                else:
                    ys.append(np.nan)
            elif percentile:
                ys.append(doc['percentile_gt'])
            else:
                ys.append(doc['gt'])
        else:
            if percentile:
                ys.append(doc['percentile'])
            else:
                ys.append(doc['score'])

    return text, ys


def imputation(data, default_score=None):
    data = np.array(data)
    if len(data.shape) == 1:
        data = np.expand_dims(data, 1)

    cols = data.shape[1]
    if not default_score:
        default_score = np.nanmean(data, 0)
    elif not isinstance(default_score, list):
        default_score = [default_score] * cols
    print default_score

    for col in range(cols):
        idxs = np.argwhere(np.isnan(data[:,col]))
        data[idxs, col] = default_score[col]

    return data

def load_docs(development_set=0.2, annotype=DEFAULT_ANNOTYPE, scoretype=DEFAULT_SCORETYPE):
    max_sents=100
    docs = []
    docs_raw = []
    with open(DATASET_PROB) as fin:
        for line in fin:
            doc = {}
            item = json.loads(line)
            docs_raw.append(item)
            parsed_text = item['parsed_text']

            sents = parsed_text['sents']

            text = ''
            pos = ''
            for sent in sents:
                sent_text = ' '.join([token[0] for token in sent['tokens']])
                sent_pos = ' '.join([token[1] for token in sent['tokens']])
                text += sent_text.strip() + ' '
                pos += sent_pos.strip() + ' '

            if len(text) < 1:
                text = item['text']

            doc['text'] = text
            doc['pos'] = pos

            if annotype == 'multitask':
                doc['score'] = []
                doc['gt'] = []
                for annotype_iter in ANNOTYPES:
                    doc['score'].append(item[annotype_iter+'_'+scoretype])
                    doc['gt'].append(item[annotype_iter+'_'+scoretype+'_'+'gt'])

            elif annotype in ANNOTYPES:
                doc['score'] = item[annotype+'_'+scoretype]
                doc['gt'] = item[annotype+'_'+scoretype+'_'+'gt']

            else:
                raise 'To be implementated'

            doc['docid'] = item['docid']
            docs.append(doc)

    docs = calculate_percentiles(docs)
    docs = dict(zip([d['docid'] for d in docs], docs))

    train_docids, dev_docids, test_docids = split_train_test(
            docs_raw, development_set=development_set)

    train_docids = [i for i in train_docids if docs[i]['score']]
    dev_docids = [i for i in dev_docids if docs[i]['score']]
    test_docids = [i for i in test_docids if docs[i]['gt'] and docs[i]['score']]
    return docs, train_docids, dev_docids, test_docids


def load_text_and_y(docs, docids, gt=False):
    docs_f = [docs[i] for i in docids]

    if gt:
        docs_f = calculate_percentiles(docs_f, field='gt', new_field='percentile_gt')

    text, y = extract_text(docs_f, gt=gt)
    return text, y



if __name__ == '__main__':
    a = np.array([[1,2,3],[1,np.nan, np.nan], [3,4.1,np.nan]])
    a = imputation(a)
    print a
    #items = load_dataset()
    #for item in items:
    #    print len(item)
