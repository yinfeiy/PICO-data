import os
import json
import numpy as np
import random
import nltk

random.seed(10)

ANNOTYPES = ['Participants', 'Intervention', 'Outcome']
DATA_PATH = '/mnt/data/workspace/nlp/PICO-data/src/tmp/dataset/'

class ExperimentReader:

    def __init__(self, annotype, binary=False):
        self._annotype = annotype
        self._load_docs()
        self._binary=binary

    def _load_docs(self):
        fn_train = [DATA_PATH + "{0}_sentence_train.data".format(self._annotype)]
        self.train_texts, self.train_labels, self.train_ids = self._load_docs_from_file(fn_train)

        fn_dev = [DATA_PATH + "{0}_sentence_dev_train.data".format(self._annotype)]
        self.dev_texts, self.dev_labels, self.dev_ids = self._load_docs_from_file(fn_dev)

        fn_test = [
                DATA_PATH + "{0}_sentence_dev_gt.data".format(self._annotype),
                DATA_PATH + "{0}_sentence_test_gt.data".format(self._annotype)
                ]
        self.test_texts, self.test_labels, self.test_ids = self._load_docs_from_file(fn_test)


    def _load_docs_from_file(self, filenames):
        texts = []
        labels = []
        ids = []

        uniqe_ids = set()
        for filename in filenames:
            with open(filename) as fin:
                lines = fin.readlines()
                random.shuffle(lines)
                for line in lines:
                    item = json.loads(line)
                    id = item["sent_id"]
                    if id in uniqe_ids:
                        continue

                    score = max(0, float(item["score"]))
                    text = item["sent"]
                    text = " ".join(nltk.word_tokenize(text))

                    texts.append(text)
                    ids.append(id)
                    labels.append([score])

                    uniqe_ids.add(id)
        return texts, labels, ids


    def has_dev_set(self):
        return self._dev_ids is not None


    def get_text_and_y(self, mode, reverse_weights=False):
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

        if self._binary:
            median = np.median(y)
            binary_y = []
            for s in y:
                if s[0] >= median:
                    binary_y.append([1])
                else:
                    binary_y.append([0])
            y = binary_y

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

if __name__ == '__main__':
    reader = ExperimentReader('Intervention')
