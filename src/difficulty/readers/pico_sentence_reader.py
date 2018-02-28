import data_utils
import os
import json
import numpy as np

ANNOTYPES = ['Participants', 'Intervention', 'Outcome']
SCORETYPES = ['corr', 'prec', 'recl']
CPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tmp_data')
DATASET =  os.path.join(CPATH, 'difficulty_weighted.json')

class PICOSentenceReader:

    def __init__(self, annotype):
        self._annotype = annotype
        self._load_docs()

    def _load_docs(self):
        gt_keys = []
        for annotype_iter in ANNOTYPES:
            for scoretype in SCORETYPES:
                key = '_'.join([annotype_iter, scoretype, 'gt'])
                gt_keys.append(key)

        self.train_texts = []
        self.train_labels = []
        self.train_ids = []
        self.test_texts = []
        self.test_labels = []
        self.test_ids = []
        with open(DATASET) as fin:
            for line in fin:
                item = json.loads(line)
                docid = item['docid']

                is_train = True
                for key in gt_keys:
                    if item.get(key, None):
                        is_train = False
                        break

                for sent_idx, sent in enumerate(item['parsed_text']['sents']):
                    tokens = [t[0] for t in sent['tokens']]

                    if self._annotype == "multitask":
                        ats=ANNOTYPES
                    else:
                        ats = [self._annotype]

                    text= ' '.join(tokens).strip()
                    labels = []
                    for at in ats:
                        key = '{0}_mv_mask'.format(at)
                        if key not in sent:
                            label = 0
                        else:
                            label = sum(sent[key])
                            label = 1 if label > 0 else 0

                        labels.append(label)

                    sent_id = '{0}_{1}'.format(docid, sent_idx)

                    if is_train:
                        self.train_texts.append(text)
                        self.train_labels.append(labels)
                        self.train_ids.append(sent_id)
                    else:
                        self.test_texts.append(text)
                        self.test_labels.append(labels)
                        self.test_ids.append(sent_id)
        dev_portion = 0.1
        num_train = len(self.train_labels)
        th = int(dev_portion*num_train)
        self.dev_texts = self.train_texts[:th]
        self.dev_labels = self.train_labels[:th]
        self.dev_ids = self.train_ids[:th]

        self.train_texts = self.train_texts[th:]
        self.train_labels = self.train_labels[th:]
        self.train_ids = self.train_ids[th:]


    def get_text_and_y(self, mode, binary=True, reverse_weights=False):
        # Ignore binary, reverse_weights
        if mode == 'train':
            text, y = self.train_texts, np.array(self.train_labels)
            ws = np.ones(y.shape)

        elif mode == 'dev':
            text, y = self.dev_texts, np.array(self.dev_labels)
            ws = np.ones(y.shape)

        elif mode == 'test':
            text, y = self.test_texts, np.array(self.test_labels)
            ws = np.ones(y.shape)
        else:
            raise "Error, mode %s is not supported", mode

        #y = data_utils.imputation(y)
        #ws = data_utils.imputation(ws, default_score=0.0)

        return text, y, ws

    def get_docids(self, mode):
        if mode == 'train':
            return self.train_ids
        elif mode == 'dev':
            return self.dev_ids
        elif mode == 'test':
            return self.test_ids
        else:
            raise "Error, mode %s is not supported", mode

if __name__ == "__main__":
    reader = PICOSentenceReader("Outcome")
