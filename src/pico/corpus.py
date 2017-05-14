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
        self.markups = self.offset2markups(markups_offset)  ## markups on token level

        self.groudtruth_offset = None
        self.groudtruth = None


    def offset2markups(self, markups_offset):
        markups = dict()

        offset2token_map = [0]*len(self.spacydoc.text)
        for i in range(self.ntokens):
            token = self.spacydoc[i]
            for j in range(len(token)):
                offset2token_map[token.idx + j] = i

        for annotype in markups_offset:
            markups[annotype] = {}

            for wid in markups_offset[annotype]:
                markups[annotype][wid] = []

                for offset_span in markups_offset[annotype][wid]:
                    span = [offset2token_map[offset_span[0]], offset2token_map[offset_span[1]-1]+1]

                    markups[annotype][wid].append(span)

        return markups


    def get_markups(self, annotype=None):
        if annotype == None:
            return self.markups
        elif annotype not in self.markups:
            return dict()
        else:
            return self.markups[annotype]


    def set_groundtruth(self, gt_markups_offset):
        self.groudtruth_offset = gt_markups_offset
        self.groudtruth = {}

        ## Combining groundtruth from multiple professionals
        markups = self.offset2markups(gt_markups_offset)
        for annotype in markups:
            mask = [0] * self.ntokens

            for wid, spans in markups[annotype].items():
                for span in spans:
                    for i in range(span[0], span[1]):
                        mask[i] = 1

            self.groudtruth[annotype] = self._mask2spans(mask)


    def text(self):
        return self.spacydoc.text


    def _mask2spans(self, mask):
        mask.append(0)  # append a non span

        spans = []
        if mask[0] == 1:
            sidx = 0

        for idx, v in enumerate(mask[1:], 1):
            if v==1 and mask[idx-1] == 0: # start of span
                sidx = idx
            elif v==0 and mask[idx-1] == 1 : # end of span
                eidx = idx
                spans.append( (sidx, eidx) )
        return spans


class Corpus:

    def __init__(self, doc_path):
        self.docs = dict()
        self.doc_path = doc_path


    def __len__(self):
        return len(self.docs)


    def load_annotations(self, annos_fn, demo_mode=False):
        with open(annos_fn) as fin:
            idx = 0
            for line in fin:
                idx += 1
                if idx % 100 == 0:
                    print '[INFO] {0} docs has been loaded'.format(idx)

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


    def load_groudtruth(self, gt_fn):
        """
        Load groudtruth for corpus, has to been called after load annotation
        """
        with open(gt_fn) as fin:
            for line in fin:
                anno = json.loads(line.strip())
                docid = anno['docid']
                del anno['docid']

                if docid not in self.docs:
                    print '[WARN] doc {0} is not loaded yet'.format(docid)
                    continue

                self.docs[docid].set_groundtruth(anno)


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

