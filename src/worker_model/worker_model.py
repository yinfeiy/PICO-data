import os
import json
import numpy as np


worker_model_path = os.path.join(
    os.path.dirname(__file__), '../../worker_quality/')

def load_worker_model(annotype):
    model_fname = os.path.join(worker_model_path, '{0}_worker_quality.json'.format(annotype))
    model = json.loads(open(model_fname).read())

    return get_worker_scores(model)

def get_worker_scores(model=None, annotype=None):

    if not model and not annotype:
        print 'Error, model and annotype cannot be None together'
        exit(1)

    worker_ids = model.keys()

    scores = {}
    for wid in worker_ids:
        wm = model[wid]
        acc = (wm[0][0] + wm[1][1]) / np.sum(wm)
        prec = wm[1][1] / (wm[1][1] + wm[0][1])
        recl = wm[1][1] / (wm[1][1] + wm[1][0])
        f1 = 2*prec*recl / (prec+recl)

        # Ignore acc, prec, recl and return f1 only
        scores[wid] = f1

    return scores

if __name__ == '__main__':
    for annotype in ['Participants', 'Intervention', 'Outcome']:
        model = load_worker_model(annotype)
        print model
        hists = np.histogram(model.values(), range=(0,1))
        print hists[0]*1.0 / np.sum(hists[0])
