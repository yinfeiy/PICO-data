import os, sys, time, datetime
import json
import random
import numpy as np

import gensim
from text_cnn import TextCNN
from tensorflow.contrib import learn
from tensorflow.contrib.tensorboard.plugins import projector

import tensorflow as tf

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularizaion lambda (default: 0.1)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS

def load_dataset():
    ifn = "difficulty_annotated.json"
    docs = []
    with open(ifn) as fin:
        for line in fin:
            item = json.loads(line)
            docs.append(item)
    return docs

target_annotype = 'Participants'
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)

    sess = tf.Session(config=session_conf)
    with sess.as_default():

        def test_step(cnn, x_batch, y_batch, writer=None):
            """
            Evaluates model on a test set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            loss, accuracy, precision, recall = sess.run(
                [cnn.loss, cnn.accuracy, cnn.precision, cnn.recall],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: loss {:g}, acc {:g}, prec {:g}, recl {:g}".format(time_str, loss, accuracy, precision, recall))

        docs = load_dataset()

        model_dir = "/mnt/data/workspace/nlp/PICO-data/src/difficulty/annotation_detector_models/model_2/"
        checkpoint = model_dir + "model-6000"
        vocab_fname = model_dir + 'vocab'

        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_fname)

        cnn=TextCNN(sequence_length = 254,
                num_classes = 2,
                vocab_size = len(vocab_processor.vocabulary_),
                embedding_size = FLAGS.embedding_dim,
                filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda = FLAGS.l2_reg_lambda)

        saver = tf.train.Saver()#tf.global_variables())
        saver.restore(sess, checkpoint)

        fout = open('difficulty_annotated_2.json', 'w+')
        for doc in docs:
            texts = []
            ys = []
            for sent in doc['parsed_text']['sents']:
                tokens = [t[0] for t in sent['tokens']]

                key = '{0}_mv_mask'.format(target_annotype)
                label = sum(sent[key])
                text= ' '.join(tokens).strip()
                label = [0, 1] if label > 0 else [1, 0]

                texts.append(text)
                ys.append(label)

            xs = np.array(list(vocab_processor.transform(texts)))
            ys = np.array(ys)

            feed_dict = {
              cnn.input_x: xs,
              cnn.input_y: ys,
              cnn.dropout_keep_prob: 1.0
            }
            dts, accuracy, precision, recall = sess.run(
                [cnn.predictions, cnn.accuracy, cnn.precision, cnn.recall], feed_dict)
            gts = np.argmax(ys,1)

            for i in range(len(doc['parsed_text']['sents'])):
                doc['parsed_text']['sents'][i]['{0}_anno_dt'.format(target_annotype)] = dts[i]
                doc['parsed_text']['sents'][i]['{0}_anno_gt'.format(target_annotype)] = gts[i]
            fout.write(json.dumps(doc) + '\n')
        fout.close()
