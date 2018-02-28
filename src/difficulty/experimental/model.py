## Local packages
import sys
sys.path.insert(0, '../')

import data_utils
import features

import features_experimental

## Global packages
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import LinearSVR, LinearSVC, SVC
from tensorflow.contrib import learn

import os
import numpy as np
import sklearn.metrics as metrics


class DifficultyModel:

    def __init__(self, classifier='SVM', annotype='Participants'):
        self.docs, self.train_docids, self.dev_docids, self.test_docids = data_utils.load_docs(annotype=annotype)

        self.annotype = annotype
        self.classifier = classifier
        self.model = None


    def prepare_svm_task(self):
        print ('Building features...')

        # Text and y
        train_text, y_train = data_utils.load_text_and_y(self.docs, self.train_docids)
        dev_text, y_dev = data_utils.load_text_and_y(self.docs, self.dev_docids)
        test_text, y_test = data_utils.load_text_and_y(self.docs, self.test_docids, gt=True)

        train_pos = data_utils.extract_pos(self.docs, self.train_docids)
        dev_pos = data_utils.extract_pos(self.docs, self.dev_docids)
        test_pos = data_utils.extract_pos(self.docs, self.test_docids)

        # NGram feature
        ngram_vectorizer = TfidfVectorizer(max_features=20000,
                                 ngram_range=(1, 3), stop_words=None, min_df=3,
                                 lowercase=True, analyzer='word')
        ngram_x_train = ngram_vectorizer.fit_transform(train_text).toarray()
        ngram_x_dev = ngram_vectorizer.transform(dev_text).toarray()
        ngram_x_test = ngram_vectorizer.transform(test_text).toarray()

        # POS feature
        pos_vectorizer = TfidfVectorizer(max_features=10000,
                                 ngram_range=(1, 3), stop_words=None, min_df=3,
                                 lowercase=True, analyzer='word')
        pos_x_train = pos_vectorizer.fit_transform(train_pos).toarray()
        pos_x_dev = pos_vectorizer.transform(dev_pos).toarray()
        pos_x_test = pos_vectorizer.transform(test_pos).toarray()

        # vocab feature (domain depande nt)
        fn_dict = os.path.join(*[os.path.dirname(__file__), '../anno_dict',
            '{0}_vocab.dict'.format(self.annotype)])
        vocab_dict = features.loadVocab(fn_dict, min_df = 5)
        vocab_x_train = features.extractBOWtFeature(self.docs, self.train_docids,
                vocab_dict, binary=False, lower=True)
        vocab_x_dev   = features.extractBOWtFeature(self.docs, self.dev_docids,
                vocab_dict, binary=False, lower=True)
        vocab_x_test  = features.extractBOWtFeature(self.docs, self.test_docids,
                vocab_dict, binary=False, lower=True)

        # Meta feature
        meta_x_train = features.extractMetaFeature(self.docs, self.train_docids)
        meta_x_dev   = features.extractMetaFeature(self.docs, self.dev_docids)
        meta_x_test  = features.extractMetaFeature(self.docs, self.test_docids)

        # deep feature
        agg = "max"
        deep_x_train = features_experimental.loadDeepFeature(self.train_docids, agg)
        deep_x_dev = features_experimental.loadDeepFeature(self.dev_docids, agg)
        deep_x_test = features_experimental.loadDeepFeature(self.test_docids, agg)

        self.x_train = np.hstack([meta_x_train
            ,deep_x_train
            #,ngram_x_train
            #,pos_x_train
            #,vocab_x_train
            ])
        self.x_dev = np.hstack([meta_x_dev
            ,deep_x_dev
            #,ngram_x_dev
            #,pos_x_dev
            #,vocab_x_dev
            ])
        self.x_test = np.hstack([meta_x_test
            ,deep_x_test
            #,ngram_x_test
            #,pos_x_test
            #,vocab_x_test
            ])

        # Filter by binary label
        self.x_train, y_train, _ = data_utils.percentile_to_binary(self.x_train, y_train, [1]*len(y_train))
        self.x_dev, y_dev, _ = data_utils.percentile_to_binary(self.x_dev, y_dev, [1]*len(y_dev))
        self.x_test, y_test, _ = data_utils.percentile_to_binary(self.x_test, y_test, [1]*len(y_test),
                lo_th=0.2, hi_th=0.8)

        # Ground Truth
        self.y_train = y_train
        self.y_dev = y_dev
        self.y_test = y_test

        print ('Building features done.')
        #self.model = LinearSVR(epsilon=0.1, C=0.1, loss='epsilon_insensitive', random_state=42)
        self.model = LinearSVC(C=10, loss='hinge', max_iter=10000, random_state=113)
        #self.model = SVC(C=1.0, kernel='rbf')

    def train(self):
        if self.classifier == 'SVM':
            if self.annotype == 'multitask':
                print 'SVM does not support multitask'
                exit()
            self.prepare_svm_task()
            self.model.fit(self.x_train, self.y_train)

            self.eval(self.x_train, self.y_train, msg="Training metrics")
            self.eval(self.x_dev, self.y_dev, msg="Development metrics")
            self.eval(self.x_test, self.y_test, msg="Testing metrics")

    def eval(self, x, y, msg=None):
        if self.classifier == 'SVM':
            y_pred = self.model.predict(x)

            if False:
                pearsonr, _ = stats.pearsonr(y, y_pred)
                spearmanr, _ = stats.spearmanr(y, y_pred)
                if msg: print msg
                print round(pearsonr, 3), round(spearmanr, 3)
            else:
                acc = metrics.accuracy_score(y, y_pred)
                if msg: print msg
                print round(acc, 3)


    def save(self):
        pass


    def load(self):
        pass


if __name__ == '__main__':
    model = DifficultyModel(classifier='SVM', annotype='Intervention')
    model.train()
