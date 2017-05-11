import json
import glob
from spacy.en import English

sp = English()
DOC_PATH = '../docs/'
ANNOTYPES = ['Participants', 'Intervention', 'Outcome']

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

    def __init__(self):
        self.docs = dict()

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
            filename = DOC_PATH + docid + '.txt'
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



if __name__ == '__main__':
    anno_path = '../annotations/'

    anno_fn = anno_path + 'PICO-annos-crowdsourcing.json'

    corpus = Corpus()
    corpus.load_annotations(anno_fn, demo_mode=True)
    corpus.load_docs()

    docid = '10036953'
    annos = corpus.get_doc_annos(docid, 'Participants')

    print annos
    print corpus.get_doc_text(docid)

    spacydoc = corpus.get_doc_spacydoc(docid)
    for wid, markups in annos.items():
        print 'Annotatison of worker', wid
        for markup in markups:
            print ' -- offset range ', spacydoc[markup[0]].idx, spacydoc[markup[1]-1].idx + spacydoc[markup[1]-1].__len__(), ': ', spacydoc[markup[0]:markup[1]]
