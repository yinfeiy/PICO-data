import tensorflow as tf
import numpy as np
import gensim
import time
import datetime
import os

from tensorflow.contrib.tensorboard.plugins import projector

from scipy import stats
from difficulty import data_utils

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 32, "Number of filters per filter size (default: 32)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1, "L2 regularizaion lambda (default: 1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

#print("\nCNN Parameters:")
#for attr, value in sorted(FLAGS.__flags.items()):
#    print("{}={}".format(attr.upper(), value))
#print("")

class CNNGraph(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):


        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedding = W
            self.embedding_no_gradient = tf.stop_gradient(self.embedding)
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

        # Calculate root mean square loss
        with tf.name_scope("loss"):
            losses = tf.square(self.scores-self.input_y)
            self.loss = tf.sqrt(tf.reduce_mean(losses)) + l2_reg_lambda * l2_loss

class CNN(object):

    def __init__(self, vocab):
        self.vocab = vocab

    def run(self, x_train, y_train, x_test, y_test):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                    allow_soft_placement=FLAGS.allow_soft_placement,
                    log_device_placement=FLAGS.log_device_placement)

            sess = tf.Session(config=session_conf)
            with sess.as_default():
                cnn=CNNGraph(sequence_length = x_train.shape[1],
                        num_classes = y_train.shape[1],
                        vocab_size = len(self.vocab.vocabulary_),
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

                # Summaries for loss
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                loss_summary = tf.summary.scalar("loss", cnn.loss)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                config_pro = projector.ProjectorConfig()
                embedding = config_pro.embeddings.add()
                embedding.tensor_name = cnn.embedding.name
                embedding.metadata_path = os.path.join(train_summary_dir, 'vocab_raw')
                projector.visualize_embeddings(train_summary_writer, config_pro)

                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables())

                # Write vocabulary
                #vocab_processor.save(os.path.join(train_summary_dir, "vocab"))
                #vks = vocab_processor.vocabulary_._reverse_mapping
                #with open(train_summary_dir + '/vocab_raw', 'w+') as fout:
                #    for v in vks:
                #        fout.write(v+'\n')

                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                ## Initialize word_embedding
                #print ('Loading w2v model...')
                #w2v_model = gensim.models.Word2Vec.load_word2vec_format('/ssd/word2vec/models/GoogleNews-vectors-negative300.bin', binary=True)
                #w2v_model = gensim.models.Word2Vec.load_word2vec_format('~/workspace/nlp/word2vec/models/vectors-reviews-restaurants.bin', binary=True)
                #print ('Load w2v model done.')

                #W_init = []
                #for v in vks:
                #    #try:
                #    #    v_vec = w2v_model[v]
                #    #except:
                #        v_vec = np.random.uniform(-1, 1, 300)
                #    W_init.append(v_vec)
                #W_init = np.array(W_init)
                #sess.run(cnn.embedding.assign(W_init))

                def train_step(x_batch, y_batch):
                    """
                    A single training step
                    """
                    feed_dict = {
                      cnn.input_x: x_batch,
                      cnn.input_y: y_batch,
                      cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    _, step, summaries, loss, y_preds = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.scores],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    if step % 1 == 0:
                        y_gt = [s[0] for s in y_batch]
                        y_dt = [s[0] for s in y_preds]
                        pearsonr, p_value = stats.pearsonr(y_gt, y_dt)
                        print("{}: step {}, loss {:g}, pearsonr {:g}".format(time_str, step, loss, pearsonr))
                    train_summary_writer.add_summary(summaries, step)

                def dev_step(x_test, y_test, writer=None):
                    """
                    Evaluates model on a dev set
                    """
                    # Generate batches
                    batches = data_utils.batch_iter(
                        list(zip(x_test, y_test)), 512, 1)

                    y_gt = []; y_dt = []
                    for batch in batches:
                        x_batch, y_batch = zip(*batch)

                        feed_dict = {
                          cnn.input_x: x_batch,
                          cnn.input_y: y_batch,
                          cnn.dropout_keep_prob: 1.0
                        }

                        step, summaries, loss, y_preds= sess.run(
                            [global_step, dev_summary_op, cnn.loss, cnn.scores],
                            feed_dict)

                        time_str = datetime.datetime.now().isoformat()
                        print("{}: step {}, loss {:g}".format(time_str, step, loss))
                        if writer:
                            writer.add_summary(summaries, step)

                        y_gt.extend([s[0] for s in y_batch])
                        y_dt.extend([s[0] for s in y_preds])
                    pearsonr, p_value = stats.pearsonr(y_gt, y_dt)
                    spearmanr, p_value = stats.spearmanr(y_gt, y_dt)

                    print(" == pearsonr {:.3g}, spearmanr {:.3g}".format(pearsonr, spearmanr))

                # Generate batches
                batches = data_utils.batch_iter(
                    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

                # Training loop. For each batch...
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        dev_step(x_test, y_test, writer=dev_summary_writer)
                        print("")
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                    if current_step % 10000 == 0:
                        break
