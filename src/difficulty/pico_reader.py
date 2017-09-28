import data_utils
import numpy as np

class PICOReader:

    def __init__(self, annotype):
        self.docs, self.train_docids, self.dev_docids, self.test_docids = data_utils.load_docs(annotype=annotype)

    def get_text_and_y(self, mode, binary=True, reverse_weights=False):
        # Text and y
        if mode == 'train':
            text, y = data_utils.load_text_and_y(self.docs, self.train_docids)
            ws = data_utils.load_weights(self.docs, self.train_docids, percentile=True, reverse=reverse_weights)

            if binary:
                text, y, ws = data_utils.percentile_to_binary(text, y, ws)
        elif mode == 'dev':
            text, y = data_utils.load_text_and_y(self.docs, self.dev_docids)
            ws = data_utils.load_weights(self.docs, self.dev_docids, percentile=True, reverse=reverse_weights)

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
