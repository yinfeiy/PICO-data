import data_utils
import numpy as np

class PICOReader:

    def __init__(self, annotype):
        self.docs, self.train_docids, self.dev_docids, self.test_docids = data_utils.load_docs(annotype=annotype)

    def get_text_and_y(self, mode, binary=True, reverse_weights=False, percentile=True):
        # Text and y
        if mode == 'train':
            text, y = data_utils.load_text_and_y(self.docs, self.train_docids)
            ws = data_utils.load_weights(self.docs, self.train_docids, percentile=percentile, reverse=reverse_weights)

            if binary:
                text, y, ws = data_utils.percentile_to_binary(text, y, ws)
        elif mode == 'dev':
            text, y = data_utils.load_text_and_y(self.docs, self.dev_docids)
            ws = data_utils.load_weights(self.docs, self.dev_docids, percentile=percentile, reverse=reverse_weights)

            if binary:
                text, y, ws = data_utils.percentile_to_binary(text, y, ws)
        elif mode == 'test':
            text, y = data_utils.load_text_and_y(self.docs, self.test_docids)
            #ws = data_utils.load_weights(self.docs, self.test_docids, percentile=True, reverse=reverse_weights)
            ws = np.ones(np.array(y).shape)

            if binary:
                text, y, ws = data_utils.percentile_to_binary(text, y, ws, lo_th=0.5, hi_th=0.5)
        else:
            raise "Error, mode %s is not supported", mode

        y = data_utils.imputation(y)
        ws = data_utils.imputation(ws, default_score=0.0)

        return text, y, ws

    def get_docids(self, mode):
        if mode == 'train':
            return self.train_docids
        elif mode == 'dev':
            return self.dev_docids
        elif mode == 'test':
            return self.test_docids

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

    def write2tsv(ofn, docids, texts, ys, ws):
        with open(ofn, 'w+') as fout:
            for did, text, y, w in zip(test_docids, texts, ys, ws):
                ostr = '{0}\t{1}\t{2}\t{3}\n'.format(did, text.replace('\t', ' '), y, w)
                fout.write(ostr)

    reader = PICOReader("Outcome")

    # train
    texts, ys, ws = reader.get_text_and_y("train", binary=False, percentile=True)
    test_docids = reader.get_docids("train")
    ys = [y[0] for y in ys]; ws = [w[0] for w in ws]

    ofn = 'tmp_data/train.tsv'
    write2tsv(ofn, test_docids, texts, ys, ws)

    # dev
    texts, ys, ws = reader.get_text_and_y("dev", binary=False, percentile=True)
    test_docids = reader.get_docids("dev")
    ys = [y[0] for y in ys]; ws = [w[0] for w in ws]

    ofn = 'tmp_data/dev.tsv'
    write2tsv(ofn, test_docids, texts, ys, ws)


    # test
    texts, ys, ws = reader.get_text_and_y("test", binary=False, percentile=True)
    test_docids = reader.get_docids("test")
    ys = [y[0] for y in ys]; ws = [w[0] for w in ws]

    ofn = 'tmp_data/test.tsv'
    write2tsv(ofn, test_docids, texts, ys, ws)

