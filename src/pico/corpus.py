import os
import json
import random
from spacy.en import English

random.seed(1395)
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
        for i in range(1, len(offset2token_map)):
            offset2token_map[i] = max(offset2token_map[i], offset2token_map[i-1])

        for annotype in markups_offset:
            markups[annotype] = {}

            for wid in markups_offset[annotype]:
                markups[annotype][wid] = []

                for offset_span in markups_offset[annotype][wid]:
                    if offset_span[1] > len(offset2token_map): # sentence boundray
                        offset_span[1] = len(offset2token_map)
                    if offset_span[1] <= offset_span[0]:       # empty span
                        continue
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


    def set_groundtruth(self, gt_markups_offset, gt_wids=None):
        self.groudtruth_offset = gt_markups_offset
        self.groudtruth = {}

        ## Combining groundtruth from multiple professionals
        markups = self.offset2markups(gt_markups_offset)
        for annotype in markups:
            mask = [0] * self.ntokens

            for wid, spans in markups[annotype].items():
                if gt_wids is not None and wid not in gt_wids:
                    continue

                for span in spans:
                    for i in range(span[0], span[1]):
                        mask[i] = 1

            self.groudtruth[annotype] = self._mask2spans(mask)

    def get_groundtruth(self, annotype):
        if annotype == None or self.groudtruth == None:
            return self.groudtruth
        elif annotype not in self.groudtruth:
            return dict()
        else:
            return self.groudtruth[annotype]

    def text(self):
        return self.spacydoc.text

    def get_markups_text(self, annotype=None):
        if annotype and annotype not in self.markups:
            return dict()

        annotypes = [annotype] if annotype else self.markups.keys()

        markups_text = {}
        for annotype in annotypes:
            markups_anno = self.markups[annotype]
            markups_text[annotype] = {}

            for wid, spans in markups_anno.iteritems():
                markups_text[annotype][wid] = []
                for span in spans:
                    text = self._get_text_by_span(span)
                    if len(text) > 0:
                        markups_text[annotype][wid].append((span, text))
        print markups_text
        if annotype:
            return markups_text[annotype]
        else:
            return markups_text

    def _get_text_by_span(self, span):
        if span[1] <= span[0]:
            return ""
        return self.spacydoc[span[0]:span[1]].text

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

    ANNOTYPES = ['Participants', 'Intervention', 'Outcome']

    def __init__(self, doc_path, verbose=True):
        self.docs = dict()
        self.doc_path = doc_path
        self.verbose = verbose

    def __len__(self):
        return len(self.docs)

    def _process_anno_per_annotype(self, anno, max_num_worker, pruned_workers):
        anno_new = {}
        wids = [ wid for wid in anno.keys() if wid not in pruned_workers ]
        wids.sort()
        random.shuffle(wids)

        if len(wids) > max_num_worker:
            wids = wids[:max_num_worker]

        for wid in wids:
            anno_new[wid] = anno[wid]

        return anno_new

    def _process_anno(self, anno, max_num_worker, pruned_workers={}):
        anno_new = {}

        max_num_worker =  1000 if max_num_worker is None else max_num_worker

        for key in anno.keys():
            if key not in self.ANNOTYPES:
                anno_new[key] = anno[key]
            else:
                annotype = key

                anno_tmp = anno[annotype]
                pruned_workers_tmp = pruned_workers.get(annotype, [])
                anno_new_tmp = self._process_anno_per_annotype(\
                        anno_tmp, max_num_worker, pruned_workers_tmp)
                anno_new[annotype] = anno_new_tmp

        return anno_new

    def load_annotations(self, annos_fn, docids=None, max_num_worker=None, pruned_workers={}):
        with open(annos_fn) as fin:
            idx = 0
            for line in fin:
                idx += 1
                if idx % 500 == 0:
                    if self.verbose:
                        print '[INFO] {0} docs has been loaded'.format(idx)

                anno = json.loads(line.strip())
                docid = anno['docid']

                if docids != None and docid not in docids: # Skip doc not in the docids parameter
                    continue

                if max_num_worker or pruned_workers:
                    anno = self._process_anno(anno, max_num_worker, pruned_workers)

                doc_fn = self.doc_path + docid + '.txt'

                del anno['docid']

                if not os.path.exists(doc_fn):
                    raise Exception('{0} not found'.format(doc_fn))

                rawdoc = open(doc_fn).read()
                spacydoc = sp(rawdoc.decode("utf8"))
                self.docs[docid] = Doc(docid, anno, spacydoc)


    def load_groudtruth(self, gt_fn, gt_wids=None):
        """
        Load groudtruth for corpus, has to been called after load annotation
        """
        with open(gt_fn) as fin:
            for line in fin:
                anno = json.loads(line.strip())
                docid = anno['docid']
                del anno['docid']

                if docid not in self.docs:
                    if self.verbose:
                        print '[WARN] doc {0} is not loaded yet'.format(docid)
                    continue

                self.docs[docid].set_groundtruth(anno, gt_wids)


    def get_doc_annos(self, docid, annotype=None, text=False):
        if docid not in self.docs:
            print 'docid {0} is not found'.format(docid)
            return None

        if text:
            return self.docs[docid].get_markups_text(annotype)
        else:
            return self.docs[docid].get_markups(annotype)


    def get_doc_groundtruth(self, docid, annotype=None):
        if docid not in self.docs:
            print 'docid {0} is not found'.format(docid)
            return None

        return self.docs[docid].get_groundtruth(annotype)

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

