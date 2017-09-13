import data_utils
import tensorflow as tf
import numpy as np

def train(model, document_reader, FLAGS):
    x_train_text, y_train = document_reader.get_text_and_y("train")
    x_train_bw_text = [ " ".join(t.split()[::-1]) for t in x_train_text ]
    x_train_l = [ max(len(text.split()), FLAGS.max_document_length) for text in x_train_text ]

    x_test_text, y_test =  document_reader.get_text_and_y("test")
    x_test_bw_text = [ " ".join(t.split()[::-1]) for t in x_test_text ]
    x_test_l = [ max(len(text.split()), FLAGS.max_document_length) for text in x_test_text ]

    with tf.Graph().as_default():
        model.Graph()

        # Basic op
        pred_op, loss_op = model.ops

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
        x_test = list(model._vocab.transform(x_test_text))

        vars = [x_train, x_train_l, y_train]
        if FLAGS.rnn_bidirectional:
            x_train_bw = list(model._vocab.transform(x_train_bw_text))
            x_test_bw = list(model._vocab.transform(x_test_bw_text))
            vars.append(x_train_bw)

        train_batches = batch_iter(list(zip(*vars)), FLAGS.batch_size, FLAGS.num_epochs)

        with tf.Session() as sess:
            sw_train = tf.summary.FileWriter(model.checkpoint_dir, sess.graph)
            sw_test = tf.summary.FileWriter(model.eval_dir, sess.graph)

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
                        x_batch, x_l_batch, y_batch, x_bw_batch= zip(*batch)
                    else:
                        x_batch, x_l_batch, y_batch =  zip(*batch)
                except ValueError:
                    continue

                feed_dict = {
                    model.input_x: x_batch,
                    model.input_l: x_l_batch,
                    model.input_y: y_batch,
                    model.input_w: np.ones((len(y_batch), len(y_batch[0]))),
                    model.dropout: FLAGS.dropout
                    }
                if FLAGS.rnn_bidirectional:
                    feed_dict[model.input_x_bw] = x_bw_batch

                sess.run(updates_op, feed_dict)
                _, step, scores, loss, train_summaries = sess.run(
                        [train_op, global_step, pred_op, loss_op, summary_op],
                        feed_dict)
                if step % 10 == 0:
                    sw_train.add_summary(train_summaries, step)


                if step % FLAGS.checkpoint_every == 0:
                    path = model.saver.save(sess, model.checkpoint_dir, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))

                if step % FLAGS.evaluate_every == 0:
                    sess.run([reset_op, table_init_op])

                    feed_dict = {
                            model.input_x: x_test,
                            model.input_l: x_test_l,
                            model.input_y: y_test,
                            model.input_w: np.ones((len(y_test), len(y_test[0]))),
                            model.dropout: 0.0
                            }
                    if FLAGS.rnn_bidirectional:
                        feed_dict[model.input_x_bw] = x_test_bw

                    sess.run(updates_op, feed_dict)
                    scores, loss, test_summaries = sess.run(
                            [pred_op, loss_op, summary_op], feed_dict)

                    sw_test.add_summary(test_summaries, step)
