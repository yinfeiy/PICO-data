import gensim
import numpy as np
import tensorflow as tf
import os

from tensorflow.contrib import learn
from tensorflow.contrib.tensorboard.plugins import projector

W2VModelFILE="/mnt/data/workspace/nlp/w2v_models/PubMed-w2v.bin"
MODEL_TRAIN = "train"

class NNModel:

    def __init__(self,
            model=MODEL_TRAIN,
            batch_size=64,
            num_classes=1,
            dropout=0.1,
            is_lstm_bidirectionral=True,
            max_document_length=64):
        self._train = True if model == MODEL_TRAIN else False
        self._dropout = dropout
        self._is_lstm_bidirectional = is_lstm_bidirectionral
        self._max_document_length = max_document_length
        self._eval_metrics = {}
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._ops = []

        self._embedding_size = 200
        self._vocab = None
        self._train_dir = './test/train/'
        self._eval_dir = './test/eval/'

    def Graph(self):
        self.input_x = tf.placeholder(tf.int32, [None, self._max_document_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self._num_classes], name="input_y")

        if self._train:
            # Assuming input text is pre-tokenized and splited by space
            vocab, init_embedding = self._LoadInitEmbeddings()

            self._vocab = learn.preprocessing.VocabularyProcessor(self._max_document_length,
                    tokenizer_fn=lambda xs:[x.split(" ") for x in xs])
            self._vocab.fit(vocab)
            print os.path.join(self._train_dir, "vocab")
            #self._vocab.save()

            vocab_size = len(self._vocab.vocabulary_)
            with tf.variable_scope("Word_Embedding"):
                embeddings = tf.get_variable(name="W", shape=init_embedding.shape,
                        initializer=tf.constant_initializer(init_embedding), trainable=False)

        else:
            self._vocab = learn.preprocessing.VocabularyProcessor.restore(os.path.join(self._train_dir, "vocab"))
            vocab_size = len(self._vocab.vocabulary_)
            with tf.variable_scope("Word_Embedding"):
                embeddings = tf.Variable(tf.constant(0, 0),
                        shape=[vocab_size, self._embedding_size],
                        trainable=False, name="W")

        input_embeddings = tf.nn.embedding_lookup(embeddings, self.input_x)
        input_embeddings = tf.Print(input_embeddings, [input_embeddings], "input_embeddings: ")
        loss = tf.reduce_sum(input_embeddings)
        self._ops.append(loss)


    def _LoadInitEmbeddings(self):
        ## Initialize word_embedding
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(W2VModelFILE, binary=True)
        vocab = []
        embd = []

        for token in w2v_model.vocab:
            vec = w2v_model[token]
            vocab.append(token)
            embd.append(vec)

        embedding = np.asarray(embd)
        return vocab, embedding

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
    model = NNModel(
            model=MODEL_TRAIN,
            dropout=FLAGS.dropout,
            is_lstm_bidirectionral=FLAGS.is_lstm_bidirectionral,
            max_document_length=FLAGS.max_document_length,
            batch_size=1)
    with tf.Session() as sess:
        model.Graph()

        sess.run(tf.global_variables_initializer())

        feed_dict = {
                model.input_x: np.array([[23,23,25]]),
                model.input_y: np.array([[1]])
                }
        ops = sess.run(model._ops, feed_dict)

if __name__ == "__main__":
    flags = tf.app.flags
    flags.DEFINE_float("dropout", 0.1, "dropout")
    flags.DEFINE_integer("max_document_length", 300, "max document length")
    flags.DEFINE_bool("is_lstm_bidirectionral", True, "whther lstm is undirectional or bidirectional")

    FLAGS = tf.flags.FLAGS
    main()
