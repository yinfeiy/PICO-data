import os
import json
from spacy.en import English

sp = English()

class Doc:

    def __init__(self, docid, markups, spacydoc=None):
        self.docid = docid
        self.spacydoc = spacydoc
        self.markups = markups

    def load_doc(self, fn):
        rawdoc = open(fn).read()
        self.spacydoc = sp(rawdoc.decode("utf8"))

    def get_markups(self, annotype=None):
        if annotype == None:
            return self.markups
        elif annotype not in self.markups:
            return dict()
        else:
            return self.markups[annotype]

    def text(self):
        return self.spacydoc.text

class Corpus:

    def __init__(self, doc_path=None):
        self.docs = dict()
        self.doc_path = doc_path

    def __len__(self):
        return len(self.docs)

    def load_annotations(self, annos_fn, demo_mode=False):
        with open(annos_fn) as fin:
            for line in fin:
                anno = json.loads(line.strip())
                docid = anno['docid']
                del anno['docid']
                self.docs[docid] = Doc(docid, anno)
                if demo_mode:
                    break       ## For demo only

    def load_docs(self):
        for docid in self.docs:
            filename = self.doc_path + docid + '.txt'
            if not os.path.exists(filename):
                raise Exception('{0} not found'.format(filename))
            self.docs[docid].load_doc(filename)

    def get_doc_annos(self, docid, annotype=None):
        if docid not in self.docs:
            print 'docid {0} is not found'.format(docid)
            return None

        if annotype != None:
            return self.docs[docid].get_markups(annotype)
        else:
            return self.docs[docid]

    def get_doc_text(self, docid):
        if docid not in self.docs:
            print 'docid {0} is not found'.format(docid)
            return None

        return self.docs[docid].text()

    def get_doc_spacydoc(self, docid):
        if docid not in self.docs:
            print 'docid {0} is not found'.format(docid)
            return None

        return self.docs[docid].spacydoc

