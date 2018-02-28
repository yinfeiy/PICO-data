from difficulty.readers import data_utils
import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics
from scipy import stats

def prepare_feed_dict(model, x, x_l, y, x_w, x_bw, dropout, rnn_bidirectional):
    feed_dict = {
            model.input_x: x,
            model.input_l: x_l,
            model.input_y: y,
            model.input_w: x_w,
            model.dropout: dropout
            }
    if rnn_bidirectional:
        feed_dict[model.input_x_bw] = x_bw
    return feed_dict


def output_preds(ofn, ids, dts, gts):
    with open(ofn, 'w+') as fout:
        for id, dt, gt in zip(ids, dts, gts):
            fout.write('{0} {1} {2}\n'.format(id, dt, gt))


def classifier_eval(labels, preds, name):
    labels = np.array(labels)
    preds = np.array(preds)

    num_tasks = labels.shape[1]
    print "{0} :".format(name)

    for i in range(num_tasks):
        labels_i = labels[:,i]
        preds_i = preds[:, i]

        acc = metrics.accuracy_score(labels_i, preds_i)
        f1 = metrics.f1_score(labels_i, preds_i)
        fpr, tpr, _ = metrics.roc_curve(labels_i, preds_i, pos_label=1)
        try:
            auc = metrics.auc(fpr, tpr)
            print "  -- Task {0}: acc: {1:.3f}, F1: {2:.3f}, AUC: {3:.3f}".format(i, acc, f1, auc)
        except:
            print "Computing metrics error, skipping."
            print zip(labels_i, preds_i)

def regressor_eval(labels, preds, name):
    labels = np.array(labels)
    preds = np.array(preds)

    num_tasks = labels.shape[1]
    print "{0} :".format(name)

    for i in range(num_tasks):
        labels_i = labels[:,i]
        preds_i = preds[:, i]
        corr, _ = stats.pearsonr(labels_i, preds_i)
        print " -- Task {0}: pearsonr: {1:.3f}".format(i, corr)


def train(model, document_reader, is_classifier, FLAGS):
    annotype = document_reader._annotype
    reverse_weights = True

    x_train_text, y_train, x_train_w = document_reader.get_text_and_y("train", reverse_weights=reverse_weights)
    x_train_bw_text = [ " ".join(t.split()[::-1]) for t in x_train_text ]
    x_train_l = [ min(len(text.split()), FLAGS.max_document_length) for text in x_train_text ]

    dev_docids = document_reader.get_docids("dev")
    x_dev_text, y_dev, x_dev_w =  document_reader.get_text_and_y("dev", reverse_weights=reverse_weights)
    x_dev_bw_text = [ " ".join(t.split()[::-1]) for t in x_dev_text ]
    x_dev_l = [ min(len(text.split()), FLAGS.max_document_length) for text in x_dev_text ]

    test_docids = document_reader.get_docids("test")
    x_test_text, y_test, x_test_w =  document_reader.get_text_and_y("test", reverse_weights=reverse_weights)
    x_test_bw_text = [ " ".join(t.split()[::-1]) for t in x_test_text ]
    x_test_l = [ min(len(text.split()), FLAGS.max_document_length) for text in x_test_text ]

    with tf.Graph().as_default():
        model.Graph()

        # Basic op
        pred_op, score_op, loss_op = model.ops

        # Train op
        reg_loss = tf.reduce_sum(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = model.loss + reg_loss

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Local init op
        reset_op = tf.local_variables_initializer()
        table_init_op = tf.tables_initializer()

        # Data preparation
        x_train = list(model._vocab.transform(x_train_text))
        x_dev = list(model._vocab.transform(x_dev_text))
        x_test = list(model._vocab.transform(x_test_text))

        vars = [x_train, x_train_l, x_train_w, y_train]
        if FLAGS.rnn_bidirectional:
            x_train_bw = list(model._vocab.transform(x_train_bw_text))
            x_dev_bw = list(model._vocab.transform(x_dev_bw_text))
            x_test_bw = list(model._vocab.transform(x_test_bw_text))
            vars.append(x_train_bw)

        train_batches = data_utils.batch_iter(list(zip(*vars)), FLAGS.batch_size, FLAGS.num_epochs)

        with tf.Session() as sess:
            #sw_train = tf.summary.FileWriter(model.checkpoint_dir, sess.graph)
            #sw_test = tf.summary.FileWriter(model.eval_dir, sess.graph)

            sess.run(tf.global_variables_initializer())

            train_updates = []
            summaries = []
            for name, (value_op, update_op) in model.eval_metrics.items():
                train_updates.append(update_op)
                summaries.append(
                        tf.summary.scalar(name, value_op))
                # tf.Print(value_op, [value_op], name)))

            summaries.append(tf.summary.scalar("model_loss", model.loss))
            summaries.append(tf.summary.scalar("reg_loss", reg_loss))
            summaries.append(tf.summary.scalar("total_loss", total_loss))

            summary_op = tf.summary.merge(summaries)
            updates_op = tf.group(*train_updates)

            for batch in train_batches:
                sess.run([reset_op, table_init_op])

                try:
                    if FLAGS.rnn_bidirectional:
                        x_batch, x_l_batch, x_w_batch, y_batch, x_bw_batch= zip(*batch)
                    else:
                        x_batch, x_l_batch, x_w_batch, y_batch =  zip(*batch)
                except ValueError:
                    continue

                feed_dict = prepare_feed_dict(model, x_batch, x_l_batch, y_batch, x_w_batch,
                        x_bw_batch, FLAGS.dropout, FLAGS.rnn_bidirectional)

                sess.run(updates_op, feed_dict)
                _, step, preds, scores, loss, train_summaries = sess.run(
                        [train_op, global_step, pred_op, score_op, loss_op, summary_op],
                        feed_dict)

                if step % 100 == 0:
                    print "Step {0}: Loss: {1:.3f}".format(step, loss)
                    if is_classifier:
                        classifier_eval(y_batch, preds, "Train")
                    else:
                        regressor_eval(y_batch, preds, "Train")
                    #sw_train.add_summary(train_summaries, step)

                if step == FLAGS.max_steps:
                    path = model.saver.save(
                            sess,
                            model.checkpoint_dir + annotype + "-model",
                            global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))
                    print("Max training steps reached. Stop training.")
                    return

                if step % FLAGS.checkpoint_every == 0:
                    path = model.saver.save(
                            sess,
                            model.checkpoint_dir + annotype + "-model",
                            global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))

                if step % FLAGS.evaluate_every == 0:
                    print "##############   EVAL   #################"
                    # Dev set
                    sess.run([reset_op, table_init_op])

                    feed_dict = prepare_feed_dict(model, x_dev, x_dev_l, y_dev, x_dev_w,
                            x_dev_bw, 0.0, FLAGS.rnn_bidirectional)

                    sess.run(updates_op, feed_dict)
                    preds, scores, loss, test_summaries = sess.run(
                            [pred_op, score_op, loss_op, summary_op], feed_dict)

                    print "Step {0}: Loss: {1:.3f}".format(step, loss)
                    if is_classifier:
                        classifier_eval(y_dev, preds, "Dev")
                    else:
                        regressor_eval(y_dev, preds, "Dev")

                    ofn = "./test/output/{0}_dev_{1}.out".format(annotype, step)
                    output_preds(ofn, dev_docids, scores, y_dev)
                    #sw_test.add_summary(test_summaries, step)

                    ## Eval set
                    sess.run([reset_op, table_init_op])

                    feed_dict = prepare_feed_dict(model, x_test, x_test_l, y_test, x_test_w,
                            x_test_bw, 0.0, FLAGS.rnn_bidirectional)

                    sess.run(updates_op, feed_dict)
                    preds, scores, loss, test_summaries = sess.run(
                            [pred_op, score_op, loss_op, summary_op], feed_dict)

                    print "Step {0}: Loss: {1:.3f}".format(step, loss)
                    if is_classifier:
                        classifier_eval(y_test, preds, "Test")
                    else:
                        regressor_eval(y_test, preds, "Test")
                    #sw_test.add_summary(test_summaries, step)

                    ofn = "./test/output/{0}_test_{1}.out".format(annotype, step)
                    output_preds(ofn, test_docids, scores, y_test)
                    print "######################################"

def eval(model, document_reader, checkpoint, is_classifier, FLAGS):
    reverse_weights = False

    train_docids = document_reader.get_docids("train")
    x_train_text, y_train, x_train_w = document_reader.get_text_and_y("train", reverse_weights=reverse_weights)
    x_train_bw_text = [ " ".join(t.split()[::-1]) for t in x_train_text ]
    x_train_l = [ min(len(text.split()), FLAGS.max_document_length) for text in x_train_text ]

    dev_docids = document_reader.get_docids("dev")
    x_dev_text, y_dev, x_dev_w =  document_reader.get_text_and_y("dev", reverse_weights=reverse_weights)
    x_dev_bw_text = [ " ".join(t.split()[::-1]) for t in x_dev_text ]
    x_dev_l = [ min(len(text.split()), FLAGS.max_document_length) for text in x_dev_text ]

    test_docids = document_reader.get_docids("test")
    x_test_text, y_test, x_test_w =  document_reader.get_text_and_y("test", reverse_weights=reverse_weights)
    x_test_bw_text = [ " ".join(t.split()[::-1]) for t in x_test_text ]
    x_test_l = [ min(len(text.split()), FLAGS.max_document_length) for text in x_test_text ]

    with tf.Graph().as_default():
        model.Graph()

        # Basic op
        pred_op, score_op, _ = model.ops

        # Data preparation
        x_train = list(model._vocab.transform(x_train_text))
        x_dev = list(model._vocab.transform(x_dev_text))
        x_test = list(model._vocab.transform(x_test_text))

        if FLAGS.rnn_bidirectional:
            x_train_bw = list(model._vocab.transform(x_train_bw_text))
            x_dev_bw = list(model._vocab.transform(x_dev_bw_text))
            x_test_bw = list(model._vocab.transform(x_test_bw_text))
        else:
            x_train_bw = x_dev_bw = x_test_bw = None

        fout = open(FLAGS.output_fname, 'w+')

        def output(ids, preds, scores, encoeded_vectors):
            for id, pred, score, vec in zip(ids, preds, scores, encoeded_vectors):
                fout.write('{0}\t{1}\t{2}\t'.format(id, pred, score))
                ostr = ""
                for fv in vec:
                    ostr += '{0} '.format(fv)
                fout.write('[{0}]\n'.format(ostr.strip()))

        with tf.Session() as sess:
            model.saver.restore(sess, checkpoint)

            ## Train Set
            batch_size = 100
            for batch_idx, i in enumerate(range(0, len(train_docids), batch_size)):
                x_ids_batch = train_docids[i:i+batch_size]
                x_batch = x_train[i:i+batch_size]
                x_l_batch = x_train_l[i:i+batch_size]
                x_w_batch = x_train_w[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                if FLAGS.rnn_bidirectional:
                    x_bw_train = x_train_bw[i:i+batch_size]

                feed_dict = prepare_feed_dict(model, x_batch, x_l_batch, y_batch, x_w_batch,
                        x_bw_train, 0.0, FLAGS.rnn_bidirectional)

                preds, scores, encoded_vector = sess.run(
                        [pred_op, score_op, model.input_encoded], feed_dict)
                classifier_eval(y_batch, preds, "Train_{0}".format(batch_idx))

                output(x_ids_batch, preds, scores, encoded_vector)

            ## Dev set
            feed_dict = prepare_feed_dict(model, x_dev, x_dev_l, y_dev, x_dev_w, x_dev_bw, 0.0,
                    FLAGS.rnn_bidirectional)
            preds, scores, encoded_vector = sess.run(
                    [pred_op, score_op, model.input_encoded], feed_dict)
            classifier_eval(y_dev, preds, "Dev")
            output(dev_docids, preds, scores, encoded_vector)

            ## Eval set
            feed_dict = prepare_feed_dict(model, x_test, x_test_l, y_test, x_test_w, x_test_bw,
                    0.0, FLAGS.rnn_bidirectional)
            preds, scores, encoded_vector = sess.run(
                    [pred_op, score_op, model.input_encoded], feed_dict)
            classifier_eval(y_test, preds, "Test")
            output(test_docids, preds, scores, encoded_vector)

        fout.close()
