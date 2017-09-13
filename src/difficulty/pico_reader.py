import data_utils

class PICOReader:

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

        y = data_utils.imputation(y)
        return text, y
