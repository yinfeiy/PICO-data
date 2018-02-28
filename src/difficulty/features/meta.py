import numpy as np

def transform(texts):
    feats = []
    for text in texts:
        text_arr = text.split()
        # Length
        num = len(text_arr)

        # number of unicode tokens
        cnt_uni = 0

        for token in text_arr:
            try:
                token.encode('ascii')
            except:
                cnt_uni += 1

        feat = [num, cnt_uni]
        feats.append(feat)
    feats = np.array(feats)
    return feats
