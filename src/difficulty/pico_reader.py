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

    reader = PICOReader("Outcome")
    texts, ys, ws = reader.get_text_and_y("test", binary=False, percentile=True)
    test_docids = reader.get_docids("test")

    ys = [ y[0] for y in ys]
    gts = dict(zip(test_docids, ys))

    import scipy.stats as stats
    ifn1 = './test/output_pos/test_200.out'
    pred_1 = {}
    with open(ifn1) as fin:
        for line in fin:
            ts = line.strip().split()
            docid = ts[0]
            score = float(ts[2])
            pred_1[docid] = score

    ifn2 = './test/output/test_200.out'
    pred_2 = {}
    with open(ifn2) as fin:
        for line in fin:
            ts = line.strip().split()
            docid = ts[0]
            score = float(ts[2])
            pred_2[docid] = score

    ss1 = []
    ss2 = []
    for docid in pred_1:
        if docid not in pred_2:
            print docid, " not in both sets"
            continue
        s_diff = abs(pred_1[docid] - pred_2[docid])
        gt = gts[docid]
        ss1.append(s_diff)
        ss2.append(gt)

    print stats.spearmanr(ss1, ss2)
