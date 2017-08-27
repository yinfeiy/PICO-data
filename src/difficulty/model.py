from cnn import CNN
try:
    from difficulty import data_utils
    from difficulty import features
except:
    import data_utils
    import features

from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import SVR, LinearSVR
from tensorflow.contrib import learn

import os
import numpy as np

class DifficultyModel:

    def __init__(self, classifier='SVM', annotype='Participants'):
        self.docs, self.train_docids, self.dev_docids, self.test_docids = data_utils.load_docs(annotype=annotype)



        #print "\n\n".join(self.train_text[:3])
        #print "\n\n".join(self.train_pos[:3])
        #exit()

        self.annotype = annotype
        self.classifier = classifier
        self.model = None


    def prepare_svm_task(self):
        print ('Building features...')

        # Text and y
        train_text, y_train = data_utils.load_text_and_y(self.docs, self.train_docids)
        dev_text, y_dev = data_utils.load_text_and_y(self.docs, self.dev_docids)
        test_text, y_test = data_utils.load_text_and_y(self.docs, self.test_docids)

        train_pos = data_utils.extract_pos(self.docs, self.train_docids)
        dev_pos = data_utils.extract_pos(self.docs, self.dev_docids)
        test_pos = data_utils.extract_pos(self.docs, self.test_docids)

        # NGram feature
        ngram_vectorizer = TfidfVectorizer(max_features=1500,
                                 ngram_range=(1, 3), stop_words=None, min_df=5,
                                 lowercase=True, analyzer='word')

        ngram_x_train = ngram_vectorizer.fit_transform(train_text).toarray()
        ngram_x_dev = ngram_vectorizer.transform(dev_text).toarray()
        ngram_x_test = ngram_vectorizer.transform(test_text).toarray()

        # POS feature
        pos_vectorizer = TfidfVectorizer(max_features=1500,
                                 ngram_range=(1, 3), stop_words=None, min_df=5,
                                 lowercase=True, analyzer='word')
        pos_x_train = pos_vectorizer.fit_transform(train_pos).toarray()
        pos_x_dev = pos_vectorizer.fit_transform(dev_pos).toarray()
        pos_x_test = pos_vectorizer.fit_transform(test_pos).toarray()

        # vocab feature (domain depande nt)
        fn_dict = os.path.join(*[os.path.dirname(__file__), 'anno_dict',
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

        self.x_train = np.hstack([meta_x_train
            #,ngram_x_train
            #,pos_x_train
            ,vocab_x_train
            ])
        self.x_dev = np.hstack([meta_x_dev
            #,ngram_x_dev
            #,pos_x_dev
            ,vocab_x_dev
            ])
        self.x_test = np.hstack([meta_x_test
            #,ngram_x_test
            #,pos_x_test
            ,vocab_x_test
            ])

        # Ground Truth
        self.y_train = y_train
        self.y_dev = y_dev
        self.y_test = y_test

        print ('Building features done.')
        self.model = LinearSVR(epsilon=0.1, C=0.1, loss='epsilon_insensitive')


    def prepare_cnn_task(self):
        # Text and y
        train_text, y_train = data_utils.load_text_and_y(self.docs, self.train_docids)
        dev_text, y_dev = data_utils.load_text_and_y(self.docs, self.dev_docids)
        test_text, y_test = data_utils.load_text_and_y(self.docs, self.test_docids)

        y_train = np.array(y_train)
        y_dev = np.array(y_dev)
        y_test = np.array(y_test)

        max_document_length = max([len(x.split(" ")) for x in train_text])
        max_document_length = int(max_document_length * 0.9);

        self.vocab = learn.preprocessing.VocabularyProcessor(max_document_length)

        self.x_train = np.array(list(self.vocab.transform(train_text)))
        self.y_train = np.expand_dims(y_train, 1) if len(y_train.shape) == 1 else y_train
        self.y_train = data_utils.imputation(y_train)

        self.x_dev = np.array(list(self.vocab.transform(dev_text)))
        self.y_dev = np.expand_dims(y_dev, 1) if len(y_dev.shape) == 1 else y_dev
        self.x_test = np.array(list(self.vocab.transform(test_text)))
        self.y_test = np.expand_dims(y_test, 1) if len(y_test.shape) == 1 else y_test
        self.y_test = data_utils.imputation(y_test)

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
    model = DifficultyModel(classifier='CNN', annotype='Outcome')
    model.train()
