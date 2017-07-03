from difficulty import data_utils

import tensorflow as tf
from tensorflow.contrib import learn

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import SVR

from scipy import stats
import numpy as np

class DifficultyModel:

    def __init__(self, classifier='SVM'):
        (self.train_text, self.y_train, self.dev_text, self.y_dev,
                self.test_text, self.y_test ) = data_utils.load_dataset(annotype='min')

        self.classifier = classifier
        self.model = None

    def prepare_svm_task(self):
        print ('Building features...')
        ngram_vectorizer = TfidfVectorizer(max_features=1500,
                                 ngram_range=(1, 3), stop_words=None, min_df=3,
                                 lowercase=False, analyzer='word')

        self.x_train = ngram_vectorizer.fit_transform(self.train_text).toarray()
        self.x_dev = ngram_vectorizer.transform(self.dev_text).toarray()
        self.x_test = ngram_vectorizer.transform(self.test_text).toarray()

    def prepare_cnn_task(self):
        max_document_length = max(
                [len(x.split(" ")) for x in self.train_text])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    def train(self):
        if self.classifier == 'SVM':
            self.prepare_svm_task()
            self.model = SVR(kernel='linear')
            self.model.fit(self.x_train, self.y_train)

            y_pred_train = self.model.predict(self.x_train)
            y_pred_dev = self.model.predict(self.x_dev)
            y_pred_test = self.model.predict(self.x_test)
            print "Training metrics:"
            print ( stats.pearsonr(self.y_train, y_pred_train),
                    stats.spearmanr(self.y_train, y_pred_train) )
            print "Development metrics:"
            print (stats.pearsonr(self.y_dev, y_pred_dev),
                    stats.spearmanr(self.y_dev, y_pred_dev) )
            print "Testing metrics:"
            print (stats.pearsonr(self.y_test, y_pred_test),
                    stats.spearmanr(self.y_test, y_pred_test) )

    def save(self):
        pass

    def load(self):
        pass

if __name__ == '__main__':
    model = DifficultyModel()
    model.train()
