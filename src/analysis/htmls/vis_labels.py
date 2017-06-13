import sys, os
sys.path.insert(0, '/mnt/data/workspace/nlp/PICO-data/src/')

from pico.corpus import Corpus, Doc
from pico import utils
import numpy as np

def viz_annotation(corpus, annotype, opath, gt=False):

    if not os.path.exists(opath):
        os.makedirs(opath)

    highlight_color = "red" if gt else "blue"

    for docid in corpus.docs:
        doc = corpus.docs[docid]
        spacydoc = doc.spacydoc
        markups = doc.markups[annotype]

        masks = np.zeros(doc.ntokens)
        for wid, spans in markups.iteritems():
            for span in spans:
                masks[span[0]:span[1]] += np.ones(span[1]-span[0])

        ostr = ''
        for idx, token in enumerate(spacydoc):
            cnt = int(masks[idx])

            if cnt > 0:
                ostr += '<font color="{2}">{0}_{1}</font> '.format(token, cnt, highlight_color)
            else:
                ostr += '{0} '.format(token)

        ostr = ostr.strip().replace('\n', '\n<br>')
        ofn = os.path.join(opath, "{0}.html".format(docid))
        with open(ofn, 'w+') as fout:
            fout.write(ostr)

def main(corpus):
    opath = 'output/'
    for annotype in utils.ANNOTYPES:
        opath_annos = '{0}/vis_annos/{1}/'.format(opath, annotype)
        opath_gts = '{0}/vis_gts/{1}/'.format(opath, annotype)

        viz_annotation(corpus, annotype, opath_annos, gt=False)
        viz_annotation(corpus, annotype, opath_gts, gt=True)

if __name__ == '__main__':
    doc_path = '../../../docs/'
    anno_fn = '../../../annotations/PICO-annos-crowdsourcing.json'
    gt_fn = '../../../annotations/PICO-annos-professional.json'


    # get pruned workers
    corpus = Corpus(doc_path = doc_path)
    corpus.load_annotations(anno_fn)
    pruned_workers = {}
    for annotype in utils.ANNOTYPES:
        pruned_workers[annotype] = utils.get_pruned_workers(corpus, annotype)

    docids = utils.docs_with_gt(gt_fn)

    corpus = Corpus(doc_path = doc_path)
    corpus.load_annotations(anno_fn, docids, pruned_workers=pruned_workers)
    corpus.load_groundtruth(gt_fn)

    main(corpus)
