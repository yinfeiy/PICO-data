import numpy as np
import os, sys
import csv
import util

pico_dir = '../PICO-annotations/'

## Participants
ANNOTYPE = 'Participants'
pop_dir = pico_dir + 'batch5k/'
popres_dir = pico_dir + 'batch5k_results/'
popres_files = ['batch5k_1to334.csv.results.1', 'batch5k_1to334.csv.results.2'
       ,'batch5k_335to1001.csv.results', 'batch5k_1002to1667.csv.results'
       ,'batch5k_internal201.csv.results'
       ]

## Intervention
#ANNOTYPE = 'Intervention'
#pop_dir = pico_dir + 'interventions_batch5k/'
#popres_dir = pico_dir + 'interventions_batch5k_results/'
#popres_files = ['batch5k_1to150.csv.results', 'batch5k_151to334.csv.results'
#                ,'batch5k_335to1001.csv.results', 'batch5k_1002to1667.csv.results'
#                #, 'batch5k_internal201.csv.results'
#                ]

## Outcome
#ANNOTYPE = 'Outcomes'
#pop_dir = pico_dir + 'outcomes_batch5k/'
#popres_dir = pico_dir + 'outcomes_batch5k_results/'
#popres_files = ['batch5k_1to150.csv.results', 'batch5k_151to334.csv.results'
#                ,'batch5k_335to1001.csv.results', 'batch5k_1002to1667.csv.results'
#                #, 'batch5k_internal201.csv.results'
#                ]

def extract_id(fn):
    pid = fn.split('.')[0]
    aid = fn.split('.')[1]
    return (pid, aid)


liste = []
def extract_spans(pop_dir, d, fn):
    f =  open( os.path.join(pop_dir, d, fn) )
    res = []
    for line in f:
        temp = line.split('\t')[1].split()
        if temp[0].strip() != ANNOTYPE: continue
        #print temp
        try:
            s = int(temp[1])
            e = int(temp[2])
            res.append((s,e))
        except ValueError:
            global liste
            liste.append(temp)
    return res

def extract_abs(pop_dir, d, fn):
    f =  open( os.path.join(pop_dir, d, fn) )
    return f.read()



res = []
dic_aid_wid = {}
for fn in popres_files:
    f = open (os.path.join(popres_dir, fn), 'r')
    reader = csv.DictReader(f, delimiter='\t', quotechar='\"')
    l = list(reader)
    res.append(l)
    for d in l:
        dic_aid_wid [ d['Answer.surveycode'] ] = d['workerid']
    #print len(l), l[0]

    f.close()

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



def make_input(tokens, pid = 0):
    """
    make input for util.build_index
    """
    s = []
    for t in tokens:
        if len(t[0]) > 0:
            s.append( t[0] + ' ' + str(pid) + '_' + str(t[1]) + '_' + str(t[2]) )
            if t [0] == '.':
                s.append ('')
    return s



def make_index(data, dic_pid):
    """
    make index for util.crowd_data
    """

    # make a list of workers
    list_wid = []
    dic_pid_data = {}
    for i, d in enumerate(data):
        list_wid.append(d[0])
        pid = d[1]
        if pid not in dic_pid_data: dic_pid_data[pid] = []
        dic_pid_data[pid].append(i)

    list_wid = sorted(list(set(list_wid)))
    dic_wid = {wid:i for i, wid in enumerate(list_wid)}
    #return (list_wid, dic_pid_data)

    # build index
    all_input = []
    for pid, s in dic_pid.items():
        tokens = tokenize(s)
        inp = make_input(tokens, pid)
        all_input.extend(inp)

    features, labels = util.build_index(all_input)

    return (list_wid, dic_wid, dic_pid_data, features, labels)


def is_inside(inv_labels, label, spans):
        if label not in inv_labels: return False
        label = inv_labels[label]
        pid, l, r = map(int, label.split('_'))
        for x, y in spans:
            if r >= x and l <= y: return True
        return False


#list of experts who provide gold labels
list_experts = ['AXQIZSZFYCA8T']

def make_crowd_data(data, dic_pid, list_wid, dic_wid, dic_pid_data, features, labels):
    """
    make util.crowd_data
    """
    #make sentences and crowdlabs


    inv_labels = {v:k for (k,v) in labels.items()}

    sentences = []
    clabs = []
    gold = []

    for pid, s in dic_pid.items():  # pubmed id with text string
        tokens = tokenize(s)
        inp = make_input(tokens, pid)
        sens = util.extract(inp, features, labels)
        if pid not in dic_pid_data:
            print pid,
            continue
        sentences.extend(sens)

        for sen in sens: # a sentence
            sen_clab = []
            for i in dic_pid_data[pid]: # a crowd annotation
                d = data[i]             # d = (wid, pid, spans)
                wlabs = [0] * len(sen)
                for j in range(len(sen)): # a word
                    if is_inside( inv_labels, sen[j].label, d[2]):
                        wlabs[j] = 1
                #if the labels is provised by expert, add it to gold
                if d[0] in list_experts:
                    gold.append((len(clabs), wlabs))
                else:
                    sen_clab.append(util.crowdlab(dic_wid[d[0]], int(d[1]), wlabs))

            clabs.append(sen_clab)


    return (util.crowd_data(sentences, clabs), gold)


def get_gold(data):
    gold = []
    for d in data:
        if d[0] in list_experts:
            gold.append(d)
    return gold

dic = {}
data = []
dic_pid = {}
def read_ann():
    cnt = 0
    global dic
    global data
    global dic_pid
    for d in sorted(os.listdir(pop_dir)):
        if os.path.isdir( os.path.join(pop_dir, d) ):

            for fn in sorted(os.listdir( os.path.join(pop_dir, d) )):
                if fn.endswith(".ann"):
                    pid, aid = extract_id(fn)
                    span = extract_spans(pop_dir, d, fn)
                    if len(span) < 1: continue
                    cnt += 1
                    dic[ (pid, aid) ] = span
                    if aid in dic_aid_wid:
                        wid = dic_aid_wid[aid]
                        data.append( (wid, pid, span) )
                elif fn.endswith(".txt"):
                    pid = fn.split('.')[0]
                    dic_pid[pid] = extract_abs(pop_dir, d, fn)

    return cnt

def main():

    (list_wid, dic_wid, dic_pid_data, features, labels) = make_index(data,
            dic_pid)

    cd, gold = make_crowd_data(data, dic_pid, list_wid, dic_wid, dic_pid_data, features,
            labels)

    return (cd, gold, list_wid, features, labels)

