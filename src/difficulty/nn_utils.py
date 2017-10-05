import data_utils
import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics

def output_preds(ofn, docids, dts, gts):
    with open(ofn, 'w+') as fout:
        for docid, gt, dt in zip(docids, dts, gts):
            fout.write('{0} {1} {2}\n'.format(docid, dt, gt))

def train(model, document_reader, FLAGS):
    reverse_weights = True

    x_train_text, y_train, x_train_w = document_reader.get_text_and_y("train", reverse_weights=reverse_weights)
    x_train_bw_text = [ " ".join(t.split()[::-1]) for t in x_train_text ]
    x_train_l = [ max(len(text.split()), FLAGS.max_document_length) for text in x_train_text ]

    dev_docids = document_reader.get_docids("dev")
    x_dev_text, y_dev, x_dev_w =  document_reader.get_text_and_y("dev", reverse_weights=reverse_weights)
    x_dev_bw_text = [ " ".join(t.split()[::-1]) for t in x_dev_text ]
    x_dev_l = [ max(len(text.split()), FLAGS.max_document_length) for text in x_dev_text ]

    test_docids = document_reader.get_docids("test")
    x_test_text, y_test, x_test_w =  document_reader.get_text_and_y("test", reverse_weights=reverse_weights)
    x_test_bw_text = [ " ".join(t.split()[::-1]) for t in x_test_text ]
    x_test_l = [ max(len(text.split()), FLAGS.max_document_length) for text in x_test_text ]

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

                feed_dict = {
                        model.input_x: x_batch,
                        model.input_l: x_l_batch,
                        model.input_y: y_batch,
                        model.input_w: x_w_batch,
                        model.dropout: FLAGS.dropout
                        }
                if FLAGS.rnn_bidirectional:
                    feed_dict[model.input_x_bw] = x_bw_batch

                sess.run(updates_op, feed_dict)
                _, step, preds, scores, loss, train_summaries = sess.run(
                        [train_op, global_step, pred_op, score_op, loss_op, summary_op],
                        feed_dict)
                if step % 10 == 0:
                    acc = metrics.accuracy_score(y_batch, preds)
                    f1 = metrics.f1_score(y_batch, preds)
                    fpr, tpr, _ = metrics.roc_curve(y_batch, scores, pos_label=1)
                    auc = metrics.auc(fpr, tpr)
                    print "Step {0}: Loss: {1:.3f} Training acc: {2:.3f}, F1: {3:.3f}, AUC: {4:.3f}".format(
                            step, loss, acc, f1, auc)
                    #sw_train.add_summary(train_summaries, step)

                if step == FLAGS.max_steps:
                    path = model.saver.save(sess, model.checkpoint_dir, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))
                    print("Max training steps reached. Stop training.")
                    return

                if step % FLAGS.checkpoint_every == 0:
                    path = model.saver.save(sess, model.checkpoint_dir, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))

                if step % FLAGS.evaluate_every == 0:
                    # Dev set
                    sess.run([reset_op, table_init_op])

                    feed_dict = {
                            model.input_x: x_dev,
                            model.input_l: x_dev_l,
                            model.input_y: y_dev,
                            model.input_w: x_dev_w,
                            model.dropout: 0.0
                            }
                    if FLAGS.rnn_bidirectional:
                        feed_dict[model.input_x_bw] = x_dev_bw

                    sess.run(updates_op, feed_dict)
                    preds, scores, loss, test_summaries = sess.run(
                            [pred_op, score_op, loss_op, summary_op], feed_dict)

                    acc = metrics.accuracy_score(y_dev, preds)
                    f1 = metrics.f1_score(y_dev, preds)
                    fpr, tpr, _ = metrics.roc_curve(y_dev, scores, pos_label=1)
                    auc = metrics.auc(fpr, tpr)
                    print "Step {0}: Loss: {1:.3f} Dev acc: {2:.3f}, F1: {3:.3f}, AUC:{4:.3f}".format(
                            step, loss, acc, f1, auc)

                    ofn = "./test/output/dev_{0}.out".format(step)
                    output_preds(ofn, dev_docids, scores, y_dev)
                    #sw_test.add_summary(test_summaries, step)

                    ## Eval set
                    sess.run([reset_op, table_init_op])

                    feed_dict = {
                            model.input_x: x_test,
                            model.input_l: x_test_l,
                            model.input_y: y_test,
                            model.input_w: x_test_w,
                            model.dropout: 0.0
                            }
                    if FLAGS.rnn_bidirectional:
                        feed_dict[model.input_x_bw] = x_test_bw

                    sess.run(updates_op, feed_dict)
                    preds, scores, loss, test_summaries = sess.run(
                            [pred_op, score_op, loss_op, summary_op], feed_dict)

                    acc = metrics.accuracy_score(y_test, preds)
                    f1 = metrics.f1_score(y_test, preds)
                    fpr, tpr, _ = metrics.roc_curve(y_test, scores, pos_label=1)
                    auc = metrics.auc(fpr, tpr)
                    print "Step {0}: Loss: {1:.3f} Eval acc: {2:.3f}, F1: {3:.3f}, AUC:{4:.3f}".format(
                            step, loss, acc, f1, auc)
                    #sw_test.add_summary(test_summaries, step)

                    ofn = "./test/output/test_{0}.out".format(step)
                    output_preds(ofn, test_docids, scores, y_test)
