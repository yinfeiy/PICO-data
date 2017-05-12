import json
import glob
from pico.corpus import Corpus, Doc

DOC_PATH = '../docs/'
ANNOTYPES = ['Participants', 'Intervention', 'Outcome']


if __name__ == '__main__':
    anno_path = '../annotations/'

    anno_fn = anno_path + 'PICO-annos-crowdsourcing.json'

    corpus = Corpus(doc_path = DOC_PATH)
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
