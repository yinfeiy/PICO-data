import os, sys, glob
import yaml

class NYTReader:
    DATA_PATH = "/mnt/data/workspace/nlp/informative/data/lead_sentence/"
    ROOT_PATH = "/mnt/data/workspace/nlp/informative/data//"

    THRESHOLD = {
            "Business": [0.33, 0.61],
            "Science": [0.23, 0.47],
            "Sports": [0.287, 0.497]
            }

    def __init__(self, genre):
        self._genre = genre
        self.docs = {}
        self.train_ids = []
        self.test_ids = []

        # load docs and train_ids, test_ids
        self.load_data()

    def get_text_and_y(self, mode):
        if mode=="train":
            return self._load_text_and_y(self.train_ids)
        elif mode =="test":
            return self._load_text_and_y(self.test_ids)
        else:
            raise "Error, mode %s is not supported.", mode

    def load_data(self):
        fn_fnames = os.path.join(
                self.ROOT_PATH, "fnames/{0}.ab.fname".format(self._genre))
        fn_scores = os.path.join(
                self.ROOT_PATH, "scores/{0}.ab.score".format(self._genre))

        fnames = open(fn_fnames).readlines()
        scores = [ float(x.split()[1]) for x in open(fn_scores).readlines() ]

        for fname, score in zip(fnames, scores):
            fname = fname.strip()
            if score >= self.THRESHOLD[self._genre][1]:
                self.docs[fname] = {"label": 1}
            elif score <= self.THRESHOLD[self._genre][0]:
                self.docs[fname] = {"label": 0}

        for fname in self.docs.keys():
            fname_full = os.path.join(self.DATA_PATH, fname.replace("txt","sentence"))
            if os.path.exists(fname_full):
                item = yaml.load(open(fname_full).read())
                if item:
                    self.docs[fname].update(item)
                #print " ".join(item["sentence_0"]["words"])

        fnames = [fn for fn in self.docs.keys() if "sentence_0" in self.docs[fn]]

        num = len(fnames)
        th = int(num*0.8)
        self.train_ids = fnames[:th]
        self.test_ids = fnames[th:]

    def _load_text_and_y(self, docids):
        texts = []
        ys = []
        for docid in docids:
            try:
                sent_ids = [key for key in self.docs[docid] if isinstance(key, str) and key.startswith("sentence_") ]
            except:
                print docid
                print self.docs[docid]
                exit()
            sent_ids.sort()
            text = ""
            for sid in sent_ids:
                sent_text = " ".join(self.docs[docid][sid]["words"])
                text = text + " " + sent_text
            texts.append(text)
            ys.append([self.docs[docid]['label']])

        return texts, ys

        return texts

if __name__ == '__main__':
    nyt_reader = NYTReader(genre="Sports")
    nyt_reader.get_text_and_y('train')
