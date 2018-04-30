import pycrfsuite
import sklearn
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import re
import json
import numpy as np
from nltk import sent_tokenize, pos_tag, word_tokenize

annotypes = ['Participants', 'Intervention', 'Outcome']
path = '/nlp/data/romap/set/'; set_path = '/nlp/data/romap/naacl-pattern/crf/joint/1/'

upper = "[A-Z]"
lower = "a-z"
punc = "[,.;:?!()]"
quote = "[\'\"]"
digit = "[0-9]"
multidot = "[.][.]"
hyphen = '[/]'
dollar = '[$]'
at = '[@]'

all_cap = upper + "+$"
all_digit = digit + "+$"

contain_cap = ".*" + upper + ".*"
contain_punc = ".*" + punc + ".*"
contain_quote = ".*" + quote + ".*"
contain_digit = ".*" + digit + ".*"
contain_multidot = ".*" + multidot + ".*"
contain_hyphen = ".*" + hyphen + ".*"
contain_dollar = ".*" + dollar + ".*"
contain_at = ".*" + at + ".*"

cap_period = upper + "\.$"


list_reg = [contain_cap, contain_punc, contain_quote,
            contain_digit, cap_period, punc, quote, digit,
            contain_multidot, contain_hyphen, contain_dollar, contain_at]

def tokenize(s):
    """
    :param s: string of the abstract
    :return: list of word with original positions
    """
    def white_char(c):
        return c.isspace() or c in [',', '?']
    res = []
    i = 0
    while i < len(s):
        while i < len(s) and white_char(s[i]): i += 1
        l = i
        while i < len(s) and (not white_char(s[i])): i += 1
        r = i
        if s[r-1] == '.':       # consider . a token
            res.append( (s[l:r-1], l, r-1) )
            res.append( (s[r-1:r], r-1, r) )
        else:
            res.append((s[l:r], l, r))
    return res

def reg_features(word, start=''):
    res = []
    for p in list_reg:
        if re.compile(p).match(word): 
            res.append(start + p + '=TRUE')
        else:
            res.append(start + p + '=FALSE')
    return res

def run(data_dict, train_docids, test_docids, fold):
    train_dict, test_dict = {}, {}
    for docid in train_docids:
        if docid in test_docids: continue
        train_dict[docid] = data_dict[docid]
        
    for docid in test_docids:
        test_dict[docid] = data_dict[docid]

    print 'Inside run'
    #list of annotated sentences over all docids
    test_sents, train_sents = [], []
    for docid in test_dict: test_sents.extend(test_dict[docid])
    for docid in train_dict: train_sents.extend(train_dict[docid])

    X_train = [sent_features(sentence) for sentence in train_sents]
    y_train = [sent_labels(sentence) for sentence in train_sents]

    #test_sents = test_sents[:10]
    X_test = [sent_features(sentence) for sentence in test_sents]
    y_test = [sent_labels(sentence) for sentence in test_sents]

    print 'Training'
    trainer = pycrfsuite.Trainer(verbose=False)
    
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
        
    trainer.set_params({'c1': 1.0,
                        'c2': 1e-3,
                        'max_iterations': 50,
                        'feature.possible_transitions': True})
    
    trainer.train(set_path + 'PICO.crfsuite')
    tagger = pycrfsuite.Tagger()
    tagger.open(set_path + 'PICO.crfsuite')

    y_pred = [tagger.tag(xseq) for xseq in X_test]
    #change filepath for output
    f = open(set_path + 'folds/fold_' + str(fold) + '.json', 'w+')
    y_dict = {'test_sents': test_sents, 'y_test': y_test, 'y_pred': y_pred}
    f.write(json.dumps(y_dict))

def all_metrics(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
        )

def mask2spans(mask):
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

def precision(spans, ref_mask):
    if len(spans) == 0: return 0
    precision_arr = []
    for span in spans:
        length = span[1]-span[0]
        poss = sum(ref_mask[span[0]:span[1]])
        precision_arr.append(1.0*poss / length)
    precision = np.mean(precision_arr)

    return precision

def recall(gold_spans, anno_mask):
    recall_arr = []
    for span in gold_spans:
        length = span[1]-span[0]
        poss = sum(anno_mask[span[0]:span[1]])
        recall_arr.append(1.0*poss / length)
    recall = np.mean(recall_arr)

    return recall

def accuracy(gold_mask, pred_mask):
    true_pos = 0
    for i in range(len(gold_mask)):
        if gold_mask[i] == pred_mask[i]: true_pos += 1
    return 1.0*true_pos/len(gold_mask)
        
def sent_features(sent):
    return [word_features(sent, i) for i in range(len(sent))]

def sent_labels(sent):
    p_labels = [token_tuple[3] for token_tuple in sent]
    i_labels = [token_tuple[4] for token_tuple in sent]
    o_labels = [token_tuple[5] for token_tuple in sent]

    all_labels = []
    for i in range(len(p_labels)):
        if p_labels[i] == 1: all_labels.append('P')
        elif i_labels[i] == 1: all_labels.append('I')
        elif o_labels[i] == 1: all_labels.append('O')
        else: all_labels.append('N')

    return all_labels

def is_disease(word, indwords):
    diseases = indwords[0]
    if word in diseases or word[:-1] in diseases:
        return True
    else:
        return False

def is_outcome(word, indwords):
    outcomes = indwords[2]
    if word in outcomes or word[:-1] in outcomes:
        return True
    else:
        return False

def is_drug(word, indwords):
    drugs = indwords[1]
    for i in range(len(drugs)):
        if drugs[i] in word:
            return True
    return False
    
def word_features(sent, i):
    sent_str = ' '.join(item[0].lower() for item in sent)
    word, postag = sent[i][0], sent[i][2]
    word_lower = word.lower()
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word.isdigit=%s' % word.isdigit(),
        'word.istitle=%s' % word.istitle(),
        'word.isupper=%s' % word.isupper(),
        'postag=' + postag,
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
                ]
    
    if i == 0: features.append('BOS')
    if i == len(sent)-1: features.append('EOS')
    
    features.extend(reg_features(word))

    L = 3
    for l in range(1, L+1, 1):
        if i-l >= 0:
            word1 = sent[i-l][0]
            postag1 = sent[i-l][2]
            features.extend([
                '-%d:word.lower=' % l + word1.lower(),
                '-%d:word.istitle=%s' % (l, word1.istitle()),
                '-%d:word.isupper=%s' % (l, word1.isupper()),
                '-%d:postag=' % l + postag1,

            ])
            features.extend(reg_features(word1, '-%d' % l))

        if i + l < len(sent):
            word1 = sent[i+l][0]
            postag1 = sent[i+l][2]
            features.extend([
                '+%d:word.lower='  % l + word1.lower(),
                '+%d:word.istitle=%s' % (l, word1.istitle()),
                '+%d:word.isupper=%s' % (l, word1.isupper()),
                '+%d:postag=' % l + postag1,
            ])
            features.extend(reg_features(word1, '+%d' % l))
            
    if i > 0:
        bigram_prev = sent[i-1][0].lower() + ' ' + word_lower
        features.append('-1:bigram=' + bigram_prev)
       
    if i > 1:
        trigram_prev = sent[i-2][0].lower() + ' ' + sent[i-1][0].lower() + ' ' + word_lower
        features.append('-1:trigram=' + trigram_prev)
       
    if i < len(sent) - 1:
        bigram_next =  word_lower + ' ' + sent[i+1][0].lower() 
        features.append('+1:bigram=' + bigram_next)
        
    if i < len(sent) - 2:
        trigram_next = word_lower + ' ' + sent[i+1][0].lower() + ' ' + sent[i+2][0].lower() 
        features.append('+1:trigram=' + trigram_next)
        
    return features


def get_train_test_sets():
    test_docids, train_docids, dev_docids, gold_docids = [], [], [], []
    
    f = open(path + 'data/docids/train.txt', 'r')
    for line in f:
        train_docids.append(line.strip())
    f = open(path + 'data/docids/test.txt', 'r')
    for line in f:
        test_docids.append(line.strip())
    f = open(path + 'data/docids/dev.txt', 'r')
    for line in f:
        dev_docids.append(line.strip())
    f = open(path + 'data/docids/gold.txt', 'r')
    for line in f:
        gold_docids.append(line.strip())
    print 'Finished loading docids'
    test_dict, train_dict, dev_dict, gold_dict = {}, {}, {}, {}
    
    
    f = open(path + 'data/annotations/HMMCrowd/training_all.json', 'r')
    for line in f:
        data = json.loads(line)

    for docid in dict:
        train_dict[docid] = dict[docid]

    print 'Finished loading data!'
    print ('Train set: ' + str(len(train_dict.keys())))
    print ('Test set: ' + str(len(gold_dict.keys())))
    
    return data, train_docids, gold_docids

if __name__ == '__main__':

    data, train_split, test_split = get_train_test_sets()


    #data = the dict file that has keys=docids, values = sentences in the specified format
    #train_split = train docids, test_split = test docids
    run(data, train_split, test_split, 0)




