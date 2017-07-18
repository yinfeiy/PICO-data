from difficulty import data_utils

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import SVR

from scipy import stats
from cnn import CNN
import numpy as np

from tensorflow.contrib import learn

class DifficultyModel:

    def __init__(self, classifier='SVM', annotype='min'):
        (self.train_text, self.y_train, self.dev_text, self.y_dev,
                self.test_text, self.y_test ) = data_utils.load_dataset(annotype=annotype, span_text=True)

        self.classifier = classifier
        self.model = None

    def prepare_svm_task(self):
        print ('Building features...')
        ngram_vectorizer = TfidfVectorizer(max_features=1500,
                                 ngram_range=(1, 3), stop_words=None, min_df=3,
                                 lowercase=False, analyzer='word')
        print ('Building features done.')

        self.x_train = ngram_vectorizer.fit_transform(self.train_text).toarray()
        self.x_dev = ngram_vectorizer.transform(self.dev_text).toarray()
        self.x_test = ngram_vectorizer.transform(self.test_text).toarray()

        self.model = SVR(kernel='linear')


    def prepare_cnn_task(self):
        max_document_length = max([len(x.split(" ")) for x in self.train_text])
        max_document_length = int(max_document_length * 0.9);

        self.vocab = learn.preprocessing.VocabularyProcessor(max_document_length)

        self.x_train = np.array(list(self.vocab.transform(self.train_text)))
        self.y_train = np.array([ [y] for y in self.y_train])
        self.x_dev = np.array(list(self.vocab.transform(self.dev_text)))
        self.y_dev = np.array([ [y] for y in self.y_dev])
        self.x_test = np.array(list(self.vocab.transform(self.test_text)))
        self.y_test = np.array([ [y] for y in self.y_test])

        self.model = CNN(self.vocab)


    def train(self):
        if self.classifier == 'SVM':
            self.prepare_svm_task()
            self.model.fit(self.x_train, self.y_train)

            self.eval(self.x_train, self.y_train, msg="Training metrics")
            self.eval(self.x_dev, self.y_dev, msg="Development metrics")
            self.eval(self.x_test, self.y_test, msg="Testing metrics")

        elif self.classifier == 'CNN':
            self.prepare_cnn_task()
            self.model.run(self.x_train, self.y_train, self.x_test, self.y_test, self.vocab)


    def eval(self, x, y, msg=None):
        if self.classifier == 'SVM':
            y_pred = self.model.predict(x)
            pearsonr, _ = stats.pearsonr(y, y_pred)
            spearmanr, _ = stats.spearmanr(y, y_pred)
            if msg: print msg
            print round(pearsonr, 3), round(spearmanr, 3)


    def save(self):
        pass


    def load(self):
        pass


if __name__ == '__main__':
    model = DifficultyModel(classifier='CNN', annotype='Participants')
    model.train()
