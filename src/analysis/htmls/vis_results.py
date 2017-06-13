import sys, os
sys.path.insert(0, '/mnt/data/workspace/nlp/PICO-data/src/')

from pico.corpus import Corpus, Doc
from pico import utils
from collections import defaultdict
import json

WORKERS = ['mv', 'dw', 'HMMCrowd']

def viz_results(corpus, annotype, opath):
    if not os.path.exists(opath):
        os.makedirs(opath)

    for docid in corpus.docs:
        spans = corpus.get_doc_annos(docid, annotype, text=True)

        ofn = os.path.join(opath, "{0}.html".format(docid))
        with open(ofn, 'w+') as fout:
            fout.write('<table sstyle="width:80%" align="center" border=1>\n')
            fout.write('<tr>\n')
            for worker in WORKERS:
                fout.write('<th>{0}</th>\n'.format(worker))
            fout.write('</tr>\n')
            fout.write('<tr>\n')
            for worker in WORKERS:
                fout.write('<td width=33%>\n')
                for span in spans.get(worker, []):
                    range, text = span
                    text = text.encode('ascii', 'ignore').decode('ascii', 'ignore')
                    fout.write(u'{0} {1}<br>'.format(range, text))
                fout.write('</td>\n')
            fout.write('</tr>\n')


def main(corpus):
    opath = 'output/'
    for annotype in utils.ANNOTYPES:
        opath_anno = '{0}/vis_res/{1}/'.format(opath, annotype)
        viz_results(corpus, annotype, opath_anno)


def merge_annos():
    methods = ['mv', 'dw', 'dw_HMM_Crowd']
    annotypes = utils.ANNOTYPES

    final_annos = {}
    for annotype  in annotypes:
        for method in methods:
            fn = '../../aggregated_results/{0}-aggregated_{1}.json'.format(annotype, method)
            with open(fn) as fin:
                for line in fin:
                    item = json.loads(line.strip())
                    docid = item['docid']
                    key = method.replace('dw_HMM_Crowd', 'HMMCrowd')
                    if docid in final_annos:
                        final_annos[docid][annotype][key] =item[annotype][key]
                    else:
                        final_annos[docid] = defaultdict(dict)
                        final_annos[docid]['docid'] = docid
                        final_annos[docid][annotype][key] = item[annotype][key]

    docids = final_annos.keys()
    docids.sort()

    ofn = 'output/tmp.json'
    with open(ofn, 'w+') as fout:
        for docid in docids:
            item = final_annos[docid]
            ostr = json.dumps(item) + '\n'
            fout.write(ostr)
    return ofn

if __name__ == '__main__':

    doc_path = '../../../docs/'

    anno_fn = merge_annos()
    gt_fn = '../../../annotations/PICO-annos-professional.json'
    gt_wids = None

    docids = utils.docs_with_gt(gt_fn)

    # Loading corpus
    corpus = Corpus(doc_path = doc_path)
    corpus.load_annotations(anno_fn, docids)

    main(corpus)

