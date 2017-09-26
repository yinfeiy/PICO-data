import os
import json
import numpy as np

worker_model_path = '../worker_quality/'

def load_worker_model(fname):
    return json.loads(open(fname).read())

def get_worker_score(model=None, annotype=None):

    if not model and not annotype:
        print 'Error, model and annotype cannot be None together'
        exit(1)

    worker_ids = model.keys()

    scores = []
    for wid in worker_ids:
        wm = model[wid]
        acc = (wm[0][0] + wm[1][1]) / np.sum(wm)
        prec = wm[1][1] / (wm[1][1] + wm[0][1])
        recl = wm[1][1] / (wm[1][1] + wm[1][0])
        f1 = 2*prec*recl / (prec+recl)

        scores.append(f1)

    return scores

if __name__ == '__main__':
    for annotype in ['Participants', 'Intervention', 'Outcome']:
        fname = os.path.join(worker_model_path, '{0}_worker_quality.json'.format(annotype))
        model = load_worker_model(fname)
        scores = get_worker_score(model=model)
        hists = np.histogram(scores, range=(0,1))
        print hists[0]*1.0 / np.sum(hists[0])
