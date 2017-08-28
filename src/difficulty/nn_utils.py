import data_utils
import tensorflow as tf
import numpy as np

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
    y_train = [[y] for y  in y_train]

    x_test_text, y_test =  document_reader.get_text_and_y("test")
    y_test = [[y] for y  in y_test]

    with tf.Graph().as_default():
        model.Graph()

        # Creat ops
        pred_op, loss_op = model.ops

        reg_loss = tf.reduce_sum(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = model.loss + reg_loss

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        reset_op = tf.local_variables_initializer()
        table_init_op = tf.tables_initializer()

        # Data preparation
        x_train = list(model._vocab.transform(x_train_text))
        x_test = list(model._vocab.transform(x_test_text))
        train_batches = data_utils.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

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

            summaries.append(tf.summary.scalar("model_loss", model.loss))
            summaries.append(tf.summary.scalar("reg_loss", reg_loss))
            summaries.append(tf.summary.scalar("total_loss", total_loss))

            summary_op = tf.summary.merge(summaries)
            updates_op = tf.group(*train_updates)

            for batch in train_batches:
                sess.run([reset_op, table_init_op])

                x_batch, y_batch = zip(*batch)
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.input_w: np.ones((len(y_batch), len(y_batch[0]))),
                    model.dropout: FLAGS.dropout
                    }

                sess.run(updates_op, feed_dict)
                _, step, scores, loss, train_summaries = sess.run(
                        [train_op, global_step, pred_op, loss_op, summary_op],
                        feed_dict)
                sw_train.add_summary(train_summaries, step)


                if step % FLAGS.checkpoint_every == 0:
                    path = model.saver.save(sess, model.checkpoint_dir, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))

                if step % FLAGS.evaluate_every == 0:
                    sess.run([reset_op, table_init_op])

                    feed_dict = {
                            model.input_x: x_test,
                            model.input_y: y_test,
                            model.input_w: np.ones((len(y_test), len(y_test[0]))),
                            model.dropout: 0.0
                            }

                    sess.run(updates_op, feed_dict)
                    scores, loss, test_summaries = sess.run(
                            [pred_op, loss_op, summary_op], feed_dict)

                    sw_test.add_summary(test_summaries, step)
