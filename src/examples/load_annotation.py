from pico.corpus import Corpus, Doc

DOC_PATH = '../docs/'
ANNOTYPES = ['Participants']


if __name__ == '__main__':
    anno_path = '../annotations/acl17-test/'

    anno_fn = anno_path + 'PICO-annos-crowdsourcing.json'
    gt_fn = anno_path + 'PICO-annos-professional.json'

    corpus = Corpus(doc_path = DOC_PATH, verbose=False)
    corpus.load_annotations(anno_fn, docids=['23549581'])
    corpus.load_groundtruth(gt_fn)

    docid = '23549581'
    annos = corpus.get_doc_annos(docid, 'Participants')

    print annos
    print corpus.get_doc_text(docid)

    spacydoc = corpus.get_doc_spacydoc(docid)
    for wid, markups in annos.items():
        print 'Annotatison of worker', wid
        for markup in markups:
            print ' -- offset range ', spacydoc[markup[0]].idx, spacydoc[markup[1]-1].idx + spacydoc[markup[1]-1].__len__(), ': ', spacydoc[markup[0]:markup[1]]
