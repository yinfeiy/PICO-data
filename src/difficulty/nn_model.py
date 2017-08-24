import data_utils
import gensim
import numpy as np
import os
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.tensorboard.plugins import projector

W2VModelFILE="/mnt/data/workspace/nlp/w2v_models/PubMed-w2v.bin"
MODE_TRAIN = "train"

class NNModel:

    def __init__(self,
            mode=MODE_TRAIN,
            encoder="CNN",
            num_classes=1,
            max_document_length=64,
            is_classifier=True,
            dropout=0.1,
            l2_reg_lambda=0.1,
            cnn_filter_sizes=[3,4,5],
            cnn_num_filters=128,
            lstm_bidirectionral=True):

        self._train = True if mode == MODE_TRAIN else False

        # Basic params
        self._max_document_length = max_document_length
        self._num_classes = num_classes
        self._is_classifier = is_classifier
        self._embedding_size = 200
        self._encoder = encoder
        self._encoding_size = 300

        # CNN params
        self._cnn_filter_sizes = cnn_filter_sizes
        self._cnn_num_filters = cnn_num_filters

        # LSTM params
        self._lstm_bidirectional = lstm_bidirectionral

        # Hyper-params
        self._l2_reg_lambda = l2_reg_lambda
        self._dropout = dropout

        self._vocab = None
        self._train_dir = './test/train/'
        self._eval_dir = './test/eval/'

        if not self._train:
            self._dropout = 0.0

        self.ops = []
        self.loss = None
        self._eval_metrics = {}


    def Graph(self):
        self.input_x = tf.placeholder(tf.int32, [None, self._max_document_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self._num_classes], name="input_y")

        if self._train:
            # Assuming input text is pre-tokenized and splited by space
            vocab, init_embedding = self._LoadInitEmbeddings()

            self._vocab = learn.preprocessing.VocabularyProcessor(self._max_document_length,
                    tokenizer_fn=lambda xs:[x.split(" ") for x in xs])
            self._vocab.fit(vocab)
            #self._vocab.save(os.path.join(self._train_dir, "vocab"))

            # Insert init embedding for <UNK>
            init_embedding = np.vstack([np.zeros(self._embedding_size)*0.1, init_embedding])

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


        if self._encoder == "CNN":
            input_encoded = self._CNNLayers(embeddings, self.input_x)

        if self._is_classifier:
            pred_scores, loss = self._classifier(input_encoded, self.input_y)
        else:
            pred_scores, loss = self._regressor(input_encoded, self.input_y)

        self.ops.extend([pred_scores, loss])
        self.loss = loss


    def _classifier(self, input_encoded, output):
        with tf.variable_scope("Classifier"):
            l2_loss = tf.constant(0.0)

            W = tf.get_variable(
                "W",
                shape=[self._encoding_size, self._num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self._num_classes]), name="b")
            scores = tf.nn.xw_plus_b(input_encoded, W, b, name="scores")

            #predictions = tf.argmax(scores, 1, name="predictions")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=output)
            logits = tf.sigmoid(scores)

            total_loss = tf.reduce_mean(losses) + self._l2_reg_lambda * l2_loss

        return logits, total_loss

    def _regressor(self, input_encoded, output):
        with tf.variable_scope("regressor"):
            total_loss = tf.constant(0.0)
            pooled_logits = []
            for idx in range(self._num_classes):
                logits = tf.layers.dense(input_encoded, 1, activation=tf.nn.sigmoid)
                gts = output[:, idx]

                pooled_logits.append(logits)
                loss = tf.nn.l2_loss(logits-gts)
                total_loss += loss

        return pooled_logits, total_loss

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


    def _LookupEmbeddings(self, embeddings, inputs):
        # Return sequence length and inputs

        mask = tf.to_float(tf.not_equal(inputs, 0))
        inputs = tf.nn.embedding_lookup(embeddings, inputs)

        lengths = tf.cast(tf.reduce_sum(mask, axis=1), tf.int64)
        return lengths, inputs


    def _CNNLayers(self, embeddings, inputs):
        _, input_embeddings = self._LookupEmbeddings(embeddings, inputs)

        input_embeddings = tf.expand_dims(input_embeddings, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self._cnn_filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Conv layer
                filter_shape = [filter_size, self._embedding_size, 1, self._cnn_num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self._cnn_num_filters]), name="b")
                conv = tf.nn.conv2d(
                        input_embeddings,
                        W,
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self._max_document_length-filter_size+1, 1, 1],
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = self._cnn_num_filters * len(self._cnn_filter_sizes)
        cnn_encoding = tf.concat(pooled_outputs, 3)
        cnn_encoding = tf.reshape(cnn_encoding, [-1, num_filters_total])

        with tf.variable_scope("dropout"):
            cnn_encoding = tf.nn.dropout(cnn_encoding, 1-self._dropout)

        cnn_encoding = tf.layers.dense(cnn_encoding, self._encoding_size)

        return cnn_encoding


    def _LSTMLayers(self):
        pass


class DocumentReader:

    def __init__(self, annotype):
        self.docs, self.train_docids, self.dev_docids, self.test_docids = data_utils.load_docs(annotype=annotype)

    def get_text_and_y(self, mode):
        # Text and y
        if mode == 'train':
            text, y = data_utils.load_text_and_y(self.docs, self.train_docids)
        elif mode == 'test':
            text, y = data_utils.load_text_and_y(self.docs, self.test_docids)
        else:
            raise "Error, mode %s is not supported", mode

        return text, y


def train(model, FLAGS):

    document_reader = DocumentReader(annotype="Participants")
    x_train_text, y_train = document_reader.get_text_and_y("train")
    #y_train = [[1] if y >0.5 else [0] for y  in y_train]
    y_train = [[y] for y  in y_train]

    with tf.Session() as sess:
        model.Graph()

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-6)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


        sess.run(tf.global_variables_initializer())

        x_train = list(model._vocab.transform(x_train_text))

        batches = data_utils.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch
                }

            ops = [train_op, global_step]
            ops.extend(model.ops)

            _, step, scores, loss = sess.run(ops, feed_dict)
            print step, loss


def eval(model):
    pass


def main():
    model = NNModel(
            mode=FLAGS.mode,
            dropout=FLAGS.dropout,
            is_classifier=False,
            max_document_length=FLAGS.max_document_length,
            encoder=FLAGS.encoder,
            cnn_filter_sizes=list(map(int, FLAGS.cnn_filter_sizes.split(","))),
            cnn_num_filters=FLAGS.cnn_num_filters,
            lstm_bidirectionral=FLAGS.lstm_bidirectionral)

    if FLAGS.mode == MODE_TRAIN:
        train(model, FLAGS)


if __name__ == "__main__":
    flags = tf.app.flags
    flags.DEFINE_string("mode", "train", "Model mode")
    flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
    flags.DEFINE_float("dropout", 0.1, "dropout")
    flags.DEFINE_integer("max_document_length", 300, "Max document length")
    flags.DEFINE_bool("lstm_bidirectionral", True, "Whther lstm is undirectional or bidirectional")
    flags.DEFINE_string("encoder", "CNN", "Type of encoder used to embed document")
    flags.DEFINE_string("cnn_filter_sizes", "3,4,5", "Filter sizes in CNN encoder")
    flags.DEFINE_integer("cnn_num_filters", 128, "Number of filters per filter size in CNN encoder")

    FLAGS = tf.flags.FLAGS
    main()
