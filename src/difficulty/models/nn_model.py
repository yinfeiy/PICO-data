from difficulty.models import nn_utils
from difficulty.readers import experiment_reader
from difficulty.readers import nyt_reader
from difficulty.readers import pico_reader
from difficulty.readers import pico_sentence_reader
import gensim
import numpy as np
import os
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.tensorboard.plugins import projector

#W2VModelFILE="/mnt/data/workspace/nlp/w2v_models/PubMed-w2v.bin"
W2VModelFILE="/mnt/data/workspace/nlp/w2v_models/PICO-w2v.vec"
EMBEDDING_DIM=200

MODE_TRAIN = "train"
MODE_EVAL = "eval"
MODE_INFER = "inference"

class NNModel:

    def __init__(self,
            mode=MODE_TRAIN,
            running_dir="./test/",
            encoder="CNN",
            num_tasks=1,
            task_names=["Task"],
            max_document_length=64,
            is_classifier=True,
            l2_reg_lambda=0.1,
            cnn_filter_sizes=[3,4,5],
            cnn_num_filters=128,
            rnn_bidirectional=False,
            rnn_cell_type="GRU",
            rnn_num_layers=2,
            dnn_layer_sizes=[512]):

        self._train = True if mode == MODE_TRAIN else False

        # Basic params
        self._max_document_length = max_document_length
        self._num_tasks = num_tasks
        self._is_classifier = is_classifier
        self._embedding_size = EMBEDDING_DIM
        self._encoder = encoder
        self._encoding_size = 300
        self._vocab = None
        self._task_names = task_names

        # CNN params
        self._cnn_filter_sizes = cnn_filter_sizes
        self._cnn_num_filters = cnn_num_filters

        # RNN params
        self._rnn_bidirectional = rnn_bidirectional
        self._rnn_cell_type = rnn_cell_type
        self._rnn_num_layers = rnn_num_layers

        # DNN params
        self._dnn_layer_sizes = dnn_layer_sizes
        self._dnn_activation = "relu"

        # Hyper-params
        self._l2_reg_lambda = l2_reg_lambda

        self.ops = []
        self.loss = None
        self.eval_metrics = {}
        self.saver = None
        self.checkpoint_dir = os.path.join(running_dir, "train/")
        self.eval_dir = os.path.join(running_dir, "test/")


    def Graph(self):
        self.input_x = tf.placeholder(tf.int32, [None, self._max_document_length], name="input_x")
        self.input_l = tf.placeholder(tf.int32, [None], name="input_l")
        self.input_y = tf.placeholder(tf.float32, [None, self._num_tasks], name="input_y")
        self.input_w = tf.placeholder(tf.float32, [None, self._num_tasks], name="input_w")
        self.dropout = tf.placeholder(tf.float32, name="dropout_prob")

        if self._rnn_bidirectional:
            self.input_x_bw = tf.placeholder(tf.int32,
                    [None, self._max_document_length], name="input_x_bw")
        else:
            self.input_x_bw = None

        # Assuming input text is pre-tokenized and splited by space
        vocab, init_embedding = self._LoadInitEmbeddings()

        def _tokenizer(xs):
            return [x.split(" ") for x in xs]
        self._vocab = learn.preprocessing.VocabularyProcessor(
                self._max_document_length, tokenizer_fn=_tokenizer)
        self._vocab.fit(vocab)

        # Insert init embedding for <UNK>
        init_embedding = np.vstack(
                [np.random.normal(size=self._embedding_size), init_embedding])

        vocab_size = len(self._vocab.vocabulary_)
        with tf.variable_scope("WordEmbeddings"):
            embeddings = tf.get_variable(name="W", shape=init_embedding.shape,
                    initializer=tf.constant_initializer(init_embedding), trainable=False)

        if self._encoder == "CNN":
            input_encoded = self._CNNLayers(embeddings)
        elif self._encoder == "RNN":
            input_encoded = self._RNNLayers(embeddings)
        elif self._encoder == "DNN":
            input_encoded = self._DNNLayers(embeddings)

        self.input_encoded = input_encoded

        with tf.variable_scope("dropout"):
            input_encoded = tf.nn.dropout(input_encoded, 1-self.dropout)

        if self._is_classifier:
            preds, pred_scores, loss = self._classifier(input_encoded, self.input_y, self.input_w)
            self.ops.extend([preds, pred_scores, loss])
        else:
            # preds and pred_scores are the same for regression model
            pred_scores, loss = self._regressor(input_encoded, self.input_y, self.input_w)
            self.ops.extend([pred_scores, pred_scores, loss])

        self.loss = loss

        self.saver = tf.train.Saver(tf.global_variables())

        return self


    def _classifier(self, input_encoded, output, weights):
        total_loss = tf.constant(0.0)
        pooled_scores = []
        pooled_predictions = []

        for idx in range(self._num_tasks):
            gts = tf.expand_dims(output[:, idx], -1)
            wts = tf.expand_dims(weights[:, idx], -1)
            with tf.variable_scope("{0}_classifier".format(self._task_names[idx])):

                labels = tf.concat([1-gts, gts], 1)
                logits = tf.layers.dense(input_encoded, 2,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(
                            self._l2_reg_lambda))

                scores = tf.reduce_max(tf.nn.softmax(logits), 1)
                predictions = tf.argmax(logits, 1, name="predictions")

                pooled_predictions.append(predictions)
                pooled_scores.append(scores)

                losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

                self.eval_metrics["{0}/Accuracy".format(self._task_names[idx])] = (
                        tf.metrics.accuracy(gts, predictions, weights=wts))
                self.eval_metrics["{0}/Precision".format(self._task_names[idx])] = (
                        tf.metrics.precision(gts, predictions, weights=wts))
                self.eval_metrics["{0}/Recall".format(self._task_names[idx])] = (
                        tf.metrics.recall(gts, predictions, weights=wts))

                total_loss += tf.reduce_mean(losses * wts)

        pooled_predictions = tf.stack(pooled_predictions, axis=1)
        pooled_scores = tf.stack(pooled_scores, axis=1)
        return pooled_predictions, pooled_scores, total_loss


    def _regressor(self, input_encoded, output, weights):
        total_loss = tf.constant(0.0)
        pooled_logits = []
        for idx in range(self._num_tasks):
            with tf.variable_scope("{0}_regressor".format(self._task_names[idx])):
                logits = tf.layers.dense(input_encoded, 1,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(
                            self._l2_reg_lambda))
                gts = tf.expand_dims(output[:, idx], -1)
                wts = tf.expand_dims(weights[:, idx], -1)

                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                        labels=gts)
                total_loss += tf.reduce_mean(losses * wts)

                pooled_logits.append(tf.sigmoid(logits))

                self.eval_metrics["{0}/Pearsonr".format(self._task_names[idx])] = (
                        tf.contrib.metrics.streaming_pearson_correlation(
                            logits, gts, weights=wts))

        pooled_logits = tf.stack(pooled_logits, axis=1)
        pooled_logits = tf.squeeze(pooled_logits, axis=-1)
        return pooled_logits, total_loss


    def _LoadInitEmbeddings(self):
        ## Initialize word_embedding
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(W2VModelFILE, binary=False)
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


    def _CNNLayers(self, embeddings):
        _, input_embeddings = self._LookupEmbeddings(embeddings, self.input_x)

        input_embeddings = tf.expand_dims(input_embeddings, -1)
        with tf.variable_scope("CNN"):
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
                cnn_encoding = tf.nn.dropout(cnn_encoding, 1-self.dropout)

            cnn_encoding = tf.layers.dense(cnn_encoding, self._encoding_size)

        return cnn_encoding


    def _DNNLayers(self, embeddings):
        lengths, input_embeddings = self._LookupEmbeddings(embeddings, self.input_x)
        lengths = tf.expand_dims(lengths, -1)
        input_embeddings = tf.divide(tf.reduce_sum(input_embeddings, 1), tf.to_float(lengths))
        #input_embeddings = tf.reduce_mean(input_embeddings, 1)
        #input_embeddings = tf.Print(input_embeddings, [input_embeddings], "input_embeddings: ", summarize=3)

        with tf.variable_scope("DNN"):
            input_tensor = tf.nn.dropout(input_embeddings, 1)

            for i, out_size in enumerate(self._dnn_layer_sizes):
                with tf.variable_scope("Layer_{0}".format(i+1)):
                    in_size = input_tensor.get_shape()[1]
                    stddev = 1.0 / tf.sqrt(tf.to_float(tf.maximum(in_size, out_size)))
                    W = tf.get_variable("W", (in_size, out_size),
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
                    b = tf.get_variable("b", (out_size),
                            initializer=tf.constant_initializer(0.1))
                    input_tensor = tf.nn.bias_add(tf.matmul(input_tensor, W), b)
                    if self._dnn_activation == "relu":
                        input_tensor = tf.nn.relu(input_tensor, name="relu")
                    else:
                        raise ValueError("dnn_activation function not supported.")

                    #if i != len(self._dnn_activation)-1:
                    #    input_tensor = tf.nn.dropout(input_tensor, 1 - self.dropout)
        #input_tensor = tf.Print(input_tensor, [input_tensor], "input_tensor: ", summarize=30)
        return input_tensor


    def _RNNCells(self):
        if self._rnn_cell_type == "GRU":
            cells= tf.contrib.rnn.MultiRNNCell(
                    [tf.nn.rnn_cell.GRUCell(self._embedding_size)
                        for x in range(self._rnn_num_layers)], state_is_tuple=True)
        elif self._rnn_cell_type == "LSTM":
            cells= tf.contrib.rnn.MultiRNNCell(
                    [tf.nn.rnn_cell.LSTMCell(self._embedding_size)
                    for x in range(self._rnn_num_layers)], state_is_tuple=True)
        return cells


    def _RNNLayers(self, embeddings):
        _, fw_embeddings = self._LookupEmbeddings(embeddings, self.input_x)

        if self._rnn_bidirectional:
            _, bw_embeddings = self._LookupEmbeddings(embeddings, self.input_x_bw)

        with tf.variable_scope("RNN"):

            with tf.variable_scope("forward"):
                fw_cells = self._RNNCells()
                _, fw_state = tf.nn.dynamic_rnn(fw_cells, fw_embeddings,
                        sequence_length=self.input_l, dtype=tf.float32)
                fw_encoding = fw_state[-1]

            if self._rnn_bidirectional:
                with tf.variable_scope("backward"):
                    bw_cells = self._RNNCells()
                    _, bw_state = tf.nn.dynamic_rnn(bw_cells, bw_embeddings,
                            sequence_length=self.input_l, dtype=tf.float32)

                    bw_encoding = bw_state[-1]
                rnn_encoding = tf.concat([fw_encoding, bw_encoding], axis=1)
            else:
                rnn_encoding = fw_encoding

            with tf.variable_scope("dropout"):
                rnn_encoding = tf.nn.dropout(rnn_encoding, 1-self.dropout)

            rnn_encoding = tf.layers.dense(rnn_encoding, self._encoding_size)

        return rnn_encoding


def main():
    #target = "PICO"
    target = "PICOSentence"
    #target = "NYT"

    if target == "PICO":
        model = NNModel(
                mode=FLAGS.mode,
                is_classifier=True,
                encoder=FLAGS.encoder,
                num_tasks=1,
                task_names=["Classification"],
                max_document_length=FLAGS.max_document_length,
                cnn_filter_sizes=list(map(int, FLAGS.cnn_filter_sizes.split(","))),
                cnn_num_filters=FLAGS.cnn_num_filters,
                rnn_bidirectional=FLAGS.rnn_bidirectional,
                rnn_cell_type=FLAGS.rnn_cell_type,
                rnn_num_layers=FLAGS.rnn_num_layers)

        document_reader = pico_reader.PICOReader(annotype="Outcome")
    elif target == "PICOSentence":
        is_classifier = False
        model = NNModel(
                mode=FLAGS.mode,
                is_classifier=is_classifier,
                encoder="CNN",
                num_tasks=1,
                task_names=["Outcome"],
                max_document_length=FLAGS.max_document_length,
                cnn_filter_sizes=list(map(int, FLAGS.cnn_filter_sizes.split(","))),
                cnn_num_filters=FLAGS.cnn_num_filters,
                rnn_bidirectional=FLAGS.rnn_bidirectional,
                rnn_cell_type=FLAGS.rnn_cell_type,
                rnn_num_layers=FLAGS.rnn_num_layers)

        #document_reader = pico_sentence_reader.PICOSentenceReader(annotype="multitask")
        document_reader = experiment_reader.ExperimentReader(annotype="Outcome", binary=is_classifier)
    elif target == "NYT":
        model = NNModel(
                mode=FLAGS.mode,
                is_classifier=True,
                encoder="CNN",
                num_tasks=1,
                task_names=["Business"],
                max_document_length=FLAGS.max_document_length,
                cnn_filter_sizes=list(map(int, FLAGS.cnn_filter_sizes.split(","))),
                cnn_num_filters=FLAGS.cnn_num_filters,
                rnn_bidirectional=FLAGS.rnn_bidirectional,
                rnn_cell_type=FLAGS.rnn_cell_type,
                rnn_num_layers=FLAGS.rnn_num_layers,
                dnn_layer_sizes=list(map(int, FLAGS.dnn_layer_sizes.split(","))))

        document_reader = nyt_reader.NYTReader(genre="Business")
    else:
        raise ValueError("Error")

    if FLAGS.mode == MODE_TRAIN:
        nn_utils.train(model, document_reader, is_classifier=is_classifier, FLAGS=FLAGS)
    elif FLAGS.mode == MODE_EVAL:
        checkpoint = "./test/train/model-2000"
        nn_utils.eval(model, document_reader, checkpoint, FLAGS=FLAGS)


if __name__ == "__main__":
    flags = tf.app.flags
    flags.DEFINE_string("mode", "train", "Model mode")
    flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    flags.DEFINE_integer("max_steps", 3000, "Max steps of training (default: 3000)")
    flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 100)")
    tf.flags.DEFINE_integer("evaluate_every", 500,
        "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 2000,
        "Save model after this many steps (default: 1000)")
    flags.DEFINE_float("dropout", 0.4, "dropout")
    flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
    flags.DEFINE_integer("max_document_length", 50, "Max document length")
    flags.DEFINE_bool("rnn_bidirectional", True,
        "Whther rnn is undirectional or bidirectional")
    flags.DEFINE_string("rnn_cell_type", "GRU", "RNN cell type, GRU or LSTM")
    flags.DEFINE_integer("rnn_num_layers", 2, "Number of layers of RNN")
    flags.DEFINE_string("encoder", "RNN", "Type of encoder used to embed document")
    flags.DEFINE_string("cnn_filter_sizes", "3,4,5", "Filter sizes in CNN encoder")
    flags.DEFINE_integer("cnn_num_filters", 32,
        "Number of filters per filter size in CNN encoder")
    flags.DEFINE_string("dnn_layer_sizes", "256", "Filter sizes in CNN encoder")
    flags.DEFINE_string("output_fname", "./tmp/output.out", "Output file")

    FLAGS = tf.flags.FLAGS
    main()
