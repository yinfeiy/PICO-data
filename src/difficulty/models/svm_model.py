from difficulty.features import readability_scores, meta, mesh_terms
from difficulty.readers.pico_abstract_reader import PICOAbstractReader

from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import LinearSVR, LinearSVC, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from tensorflow.contrib import learn
import sklearn.metrics as metrics

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

class DifficultyModel:

    def __init__(self, example_reader=None, classifier='SVM'):
        self._reader = example_reader

        self._annotype = example_reader.name()
        self._classifier = classifier
        self._model = None


    def prepare_svm_task(self):
        print ('Building features...')

        train_ids = self._reader.get_docids('train')
        dev_ids = self._reader.get_docids('dev')
        test_ids = self._reader.get_docids('test')

        # Text and y
        texts_train, ys_train, _ = self._reader.get_text_and_y('train')
        texts_dev,   ys_dev,   _ = self._reader.get_text_and_y('dev')
        texts_test,  ys_test,  _ = self._reader.get_text_and_y('test')

        # NGram feature
        ngram_vectorizer = TfidfVectorizer(max_features=20000,
                                 ngram_range=(1, 3), stop_words=None, min_df=3,
                                 lowercase=True, analyzer='word')
        ngram_x_train = ngram_vectorizer.fit_transform(texts_train).toarray()
        ngram_x_dev   = ngram_vectorizer.transform(texts_dev).toarray()
        ngram_x_test  = ngram_vectorizer.transform(texts_test).toarray()

        # Meta feature
        meta_x_train = meta.transform(texts_train)
        meta_x_dev   = meta.transform(texts_dev)
        meta_x_test  = meta.transform(texts_test)

        # Readability feature
        readability_x_train = readability_scores.transform(texts_train)
        readability_x_dev   = readability_scores.transform(texts_dev)
        readability_x_test  = readability_scores.transform(texts_test)

        # Mesh term feature
        mterm_x_train = mesh_terms.transform_by_docids(train_ids)
        mterm_x_dev   = mesh_terms.transform_by_docids(dev_ids)
        mterm_x_test  = mesh_terms.transform_by_docids(test_ids)

        self.xs_train = np.hstack([meta_x_train
            #,ngram_x_train
            #,readability_x_train
            ,mterm_x_train
            ])
        self.xs_dev = np.hstack([meta_x_dev
            #,ngram_x_dev
            #,readability_x_dev
            ,mterm_x_dev
            ])
        self.xs_test = np.hstack([meta_x_test
            #,ngram_x_test
            #,readability_x_test
            ,mterm_x_test
            ])

        # Filter by binary label
        #self.x_train, y_train = data_utils.percentile_to_binary(self.x_train, y_train)
        #self.x_dev, y_dev = data_utils.percentile_to_binary(self.x_dev, y_dev)
        #self.x_test, y_test = data_utils.percentile_to_binary(self.x_test, y_test, lo_th=0.5, hi_th=0.5)

        # Ground Truth
        self.ys_train = ys_train
        self.ys_dev   = ys_dev
        self.ys_test  = ys_test

        print ('Building features done.')
        #self.model = LinearSVR(C=1)
        self.model = LinearRegression(normalize=True)
        #self.model = LinearSVC(C=10, loss='hinge', max_iter=10000, random_state=42)
        #self.model = SVC(C=1.0, kernel='rbf')

    def train(self):
        if self._classifier == 'SVM':
            self.prepare_svm_task()
            self.model.fit(self.xs_train, self.ys_train)
            features = zip(mesh_terms.get_feature_terms(), self.model.coef_[0][2:])
            features.sort(key=lambda x:x[1])
            fout = open("mesh_terms_{0}.csv".format(self._annotype), "w+")
            for feature in features:
                fout.write("\"{0}\", {1}\n".format(feature[0], feature[1]))
            fout.close()
            self.eval(self.xs_train, self.ys_train, msg="Training metrics")
            self.eval(self.xs_dev,   self.ys_dev,   msg="Development metrics")
            self.eval(self.xs_test,  self.ys_test,  msg="Testing metrics")

    def eval(self, x, y, msg=None):
        if self._classifier == 'SVM':
            y_pred = self.model.predict(x)
            y_pred = y_pred.flatten()
            y = y.flatten()

            if True:
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
    reader = PICOAbstractReader("Intervention")
    model = DifficultyModel(classifier='SVM', example_reader=reader)
    model.train()
