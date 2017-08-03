from difficulty import data_utils

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import SVR

from scipy import stats
from cnn import CNN
import numpy as np

from tensorflow.contrib import learn

class DifficultyModel:

    def __init__(self, classifier='SVM', annotype='Participants'):
        docs, train_docids, dev_docids, test_docids = data_utils.load_docs(annotype=annotype)

        self.train_text, self.y_train = data_utils.load_text_and_y(docs, train_docids)
        self.dev_text, self.y_dev = data_utils.load_text_and_y(docs, dev_docids)
        self.test_text, self.y_test = data_utils.load_text_and_y(docs, test_docids)

        self.train_pos = data_utils.extract_pos(docs, train_docids)
        self.dev_pos = data_utils.extract_pos(docs, dev_docids)
        self.test_pos = data_utils.extract_pos(docs, test_docids)

        #print "\n\n".join(self.train_text[:3])
        #print "\n\n".join(self.train_pos[:3])
        #exit()

        self.annotype = annotype
        self.classifier = classifier
        self.model = None

    def prepare_svm_task(self):
        print ('Building features...')
        ngram_vectorizer = TfidfVectorizer(max_features=1500,
                                 ngram_range=(1, 3), stop_words=None, min_df=5,
                                 lowercase=True, analyzer='word')
        pos_vectorizer = TfidfVectorizer(max_features=1500,
                                 ngram_range=(1, 3), stop_words=None, min_df=5,
                                 lowercase=True, analyzer='word')

        self.x_train = ngram_vectorizer.fit_transform(self.train_text).toarray()
        self.x_dev = ngram_vectorizer.transform(self.dev_text).toarray()
        self.x_test = ngram_vectorizer.transform(self.test_text).toarray()

        if False:
            self.x_train = np.hstack([self.x_train,
                    pos_vectorizer.fit_transform(self.train_pos).toarray()])
            self.x_dev = np.hstack([self.x_dev,
                pos_vectorizer.transform(self.dev_pos).toarray()])
            self.x_test = np.hstack([self.x_test,
                pos_vectorizer.transform(self.test_pos).toarray()])

        print ('Building features done.')
        self.model = SVR(kernel='rbf')


    def prepare_cnn_task(self):
        max_document_length = max([len(x.split(" ")) for x in self.train_text])
        max_document_length = int(max_document_length * 0.9);

        self.vocab = learn.preprocessing.VocabularyProcessor(max_document_length)

        self.x_train = np.array(list(self.vocab.transform(self.train_text)))
        self.y_train = np.array(self.y_train)
        self.y_train = np.expand_dim(self.y_train, 1) \
                if len(self.y_train.shape) == 1 else self.y_train

        self.y_train = data_utils.imputation(self.y_train)

        self.x_dev = np.array(list(self.vocab.transform(self.dev_text)))
        self.y_dev = np.array(self.y_dev)
        self.y_dev = np.expand_dim(self.y_dev, 1) \
                if len(self.y_dev.shape) == 1 else self.y_dev
        self.x_test = np.array(list(self.vocab.transform(self.test_text)))
        self.y_test = np.array(self.y_test)
        self.y_test = np.expand_dim(self.y_test, 1) \
                if len(self.y_test.shape) == 1 else self.y_test
        self.y_test = data_utils.imputation(self.y_test)

        self.model = CNN(self.vocab)


    def train(self):
        if self.classifier == 'SVM':
            if self.annotype == 'multitask':
                print 'SVM does not support multitask'
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
    model = DifficultyModel(classifier='SVM', annotype='Outcome')
    model.train()
