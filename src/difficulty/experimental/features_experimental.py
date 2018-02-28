from collections import defaultdict
import numpy as np

FEATURE_DIMS = 300
def loadDeepFeature(docids=[], agg="mean"):
    feature_fname = '../tmp/output.out'

    doc_features = defaultdict(list)
    with open(feature_fname) as fin:
        for line in fin:
            ts = line.strip().split('\t')
            feat_vec = [float(v) for v in ts[-1].replace('[','').replace(']', '').split(' ')]
            doc_id, sent_id = ts[0].split('_')
            doc_features[doc_id].append((int(sent_id), feat_vec))

    features = []
    for docid in docids:
        if docid not in docids:
            features.append(np.zeros(FEATURE_DIMS))
        feat_vecs = np.array([f[1] for f in doc_features[docid]])
        if agg == "mean":
            feat = np.mean(feat_vecs, axis=0)
        elif agg == "max":
            feat = np.amax(feat_vecs, axis=0)
        elif agg == "combine":
            feat_1 = np.mean(feat_vecs, axis=0)
            feat_2 = np.amax(feat_vecs, axis=0)
            feat = np.hstack([feat_1, feat_2])
        else:
            raise "Error, agg type is not supported"
        features.append(feat)

    return features

if __name__ == '__main__':
    feats = loadDeepFeature(["23549581"], agg="combine")

    print np.array(feats[0]).shape
