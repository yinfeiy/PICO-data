import gensim
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.tensorboard.plugins import projector

W2VModelFILE="/mnt/data/workspace/nlp/w2v_models/PubMed-w2v.bin"
MODEL_TRAIN = "train"

class NNModel:

    def __init__(self,
            model,
            dropout,
            is_lstm_bidirectionral,
            max_document_length):
        self._train = True if model == MODEL_TRAIN else False
        self._dropout = dropout
        self._is_lstm_bidirectional = is_lstm_bidirectionral
        self._max_document_length = max_document_length
        self._eval_metrics = {}

        self._vocab = None
        self.train_dir = None
        self.eval_dir = None


    def Graph():
        self._embedding_size, self._embedding_placeholder = self._Embeddings()

        if self._train:
            # Assuming input text is pre-tokenized and splited by space
            self._vocab = learn.preprocessing.VocabularyProcessor(self._max_document_length,
                    tokenizer_fn=lambda xs:[x.split(" ") for x in xs])
                    )
            self._vocab.fit(vocab)
            self._vocab.save(os.path.join(self._train_dir, "vocab"))
        else:
            self._vocab = learn.preprocessing.VocabularyProcessor.restore(os.path.join(self._train_dir, "vocab"))



    def _LoadInitEmbeddings(self):
        ## Initialize word_embedding
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(W2VModelFILE, binary=True)
        vocab = []
        embd = []

        for token in w2v_model.vocab:
            vec = w2v_model[v]
            vocab.append(token)
            embd.append(vec)

        vocab_size = len(vocab)
        embedding_size = len(embd[0])
        embedding = np.asarray(embd)
        with tf.variable_scope("Word Embedding"):
            W = tf.Variable(tf.constant(0, 0), shape=[vocab, embedding_size],
                    tf.constant_initilizer(embedding), trainable=False, name="W")

        return embedding


    def _Embeddings(self):
            embedding_placeholder = tf.placeholer(tf.float32, [vocab_size, embedding_size])
            embedding_init = W.assign(self.embedding_placeholder)
        return embedding_size, embedding_placeholder


    def _LookupEmbeddings(self, input_seqs, embeddings):
        # Return masks and inputs
        pass


    def _CNNLayers(self):
        pass


    def _LSTMLayers(self):
        pass


def train(model):
    pass

def eval(model):
    pass

def main():
    nn_model = NNModel(
            dropout=FLAGS.dropout,
            is_lstm_bidirectionral=FLAGS.is_lstm_bidirectionral,
            max_document_length=FLAGS.max_document_length)

if __name__ == "__main__":
    flags = tf.app.flags
    flags.DEFINE_float("dropout", 0.1, "dropout")
    flags.DEFINE_init("max_document_length", 300, "max document length")
    flags.DEFINE_bool("is_lstm_bidirectionral", True, "whther lstm is undirectional or bidirectional")
