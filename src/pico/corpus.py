import os
import json
from spacy.en import English

sp = English()

class Doc:

    def __init__(self, docid, markups_offset, spacydoc=None):
        self.docid = docid
        self.spacydoc = spacydoc
        self.ntokens = len(spacydoc)

        self.markups_offset = markups_offset  ## offset markups on string character level
        self.markups = dict()                 ## markups on token level
        self.offset2markups()

    def offset2markups(self):
        offset2token_map = [0]*len(self.spacydoc.text)
        for i in range(self.ntokens):
            token = self.spacydoc[i]
            for j in range(len(token)):
                offset2token_map[token.idx + j] = i

        for annotype in self.markups_offset:
            self.markups[annotype] = {}

            for wid in self.markups_offset[annotype]:
                self.markups[annotype][wid] = []

                for offset_span in self.markups_offset[annotype][wid]:
                    span = [offset2token_map[offset_span[0]], offset2token_map[offset_span[1]-1]+1]

                    self.markups[annotype][wid].append(span)


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

    def __init__(self, doc_path):
        self.docs = dict()
        self.doc_path = doc_path

    def __len__(self):
        return len(self.docs)

    def load_annotations(self, annos_fn, demo_mode=False):
        with open(annos_fn) as fin:
            for line in fin:
                anno = json.loads(line.strip())
                docid = anno['docid']
                doc_fn = self.doc_path + docid + '.txt'

                del anno['docid']

                if not os.path.exists(doc_fn):
                    raise Exception('{0} not found'.format(doc_fn))

                rawdoc = open(doc_fn).read()
                spacydoc = sp(rawdoc.decode("utf8"))
                self.docs[docid] = Doc(docid, anno, spacydoc)

                if demo_mode:
                    break       ## For demo only


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

