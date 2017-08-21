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
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularizaion lambda (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on test set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS

def load_dataset():
    ANNOTYPES = ['Participants', 'Intervention', 'Outcome']
    SCORETYPES = ['corr', 'prec', 'recl']
    ifn = "difficulty_annotated.json"

    target_annotype = 'Participants'

    gt_keys = []
    for annotype in ANNOTYPES:
        for scoretype in SCORETYPES:
            key = '_'.join([annotype, scoretype, 'gt'])
            gt_keys.append(key)

    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []
    with open(ifn) as fin:
        for line in fin:
            item = json.loads(line)
            docid = item['docid']

            is_train = True
            for key in gt_keys:
                if item.get(key, None):
                    is_train = False
                    break

            for sent in item['parsed_text']['sents']:
                tokens = [t[0] for t in sent['tokens']]

                key = '{0}_mv_mask'.format(target_annotype)
                if key not in sent:
                    continue
                label = sum(sent[key])
                text= ' '.join(tokens).strip()
                label = [1, 0] if label > 0 else [0, 1]

                if is_train:
                    train_texts.append(text)
                    train_labels.append(label)
                else:
                    test_texts.append(text)
                    test_labels.append(label)

    return train_texts, train_labels, test_texts, test_labels

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

x_train_text, y_train, x_test_text, y_test = load_dataset()

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        def train_step(cnn, train_op, global_step, x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy, precision, recall= sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if step % 100 == 0:
                print("{}: step {}, loss {:g}, acc {:g}, prec {:g}, recl {:g}".format(time_str, step, loss, accuracy, precision, recall))
            train_summary_writer.add_summary(summaries, step)

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

        if True:
            max_document_length = max([len(x.split(" ")) for x in x_train_text])
            vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
            vocab_processor.fit(x_train_text)

            x_train = np.array(list(vocab_processor.transform(x_train_text)))
            x_test = np.array(list(vocab_processor.transform(x_test_text)))
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            cnn=TextCNN(sequence_length = x_test.shape[1],
                    num_classes = 2,
                    vocab_size = len(vocab_processor.vocabulary_),
                    embedding_size = FLAGS.embedding_dim,
                    filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda = FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_info", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            prec_summary = tf.summary.scalar("precision", cnn.precision)
            recl_summary = tf.summary.scalar("recall", cnn.recall)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, prec_summary, recl_summary, grad_summaries_merged])
            train_summary_dir = out_dir #os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            config_pro = projector.ProjectorConfig()
            embedding = config_pro.embeddings.add()
            embedding.tensor_name = cnn.embedding.name
            embedding.metadata_path = os.path.join(out_dir, 'vocab_raw')
            projector.visualize_embeddings(train_summary_writer, config_pro)

            # Dev summaries
            test_summary_op = tf.summary.merge([loss_summary, acc_summary])
            test_summary_dir = os.path.join(out_dir, "summaries", "test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = out_dir #os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver()#tf.global_variables())

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))
            vks = vocab_processor.vocabulary_._reverse_mapping
            with open(out_dir + '/vocab_raw', 'w+') as fout:
                for v in vks:
                    fout.write(v.encode("ascii","ignore").decode("ascii","ignore")+'\n')

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            ## Initialize word_embedding
            W2VModelFILE="/mnt/data/workspace/nlp/w2v_models/PubMed-w2v.bin"
            w2v_model = gensim.models.KeyedVectors.load_word2vec_format(W2VModelFILE, binary=True)
            W_init = []
            for v in vks:
                try:
                    v_vec = w2v_model[v]
                except:
                    v_vec = np.random.uniform(-1, 1, FLAGS.embedding_dim)
                W_init.append(v_vec)
            W_init = np.array(W_init)
            sess.run(cnn.embedding.assign(W_init))


            # Generate batches
            batches = batch_iter(
                    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(cnn, train_op, global_step, x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    test_step(cnn, x_test, y_test)#, writer=test_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if current_step % 10000 == 0:
                    break
        else:
            model_dir = "/mnt/data/workspace/nlp/PICO-data/src/difficulty/runs_info/1502255947/"
            checkpoint = model_dir + "model-2000"
            vocab_fname = model_dir + 'vocab'

            vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_fname)

            x_test = np.array(list(vocab_processor.transform(x_test_text)))
            y_test = np.array(y_test)

            cnn=TextCNN(sequence_length = x_test.shape[1],
                    num_classes = 2,
                    vocab_size = len(vocab_processor.vocabulary_),
                    embedding_size = FLAGS.embedding_dim,
                    filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda = FLAGS.l2_reg_lambda)

            saver = tf.train.Saver()#tf.global_variables())
            saver.restore(sess, checkpoint)
            print("\nEvaluation:")
            test_step(cnn, x_test, y_test)#, writer=test_summary_writer)
            print("")
