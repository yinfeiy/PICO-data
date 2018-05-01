import json, re
import nltk
from nltk import pos_tag, word_tokenize
annotypes = ['Participants', 'Intervention', 'Outcome']

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

def sentence_split(items):
    sentences, sentence = [], []
    for item in items:
        sentence.append(item)
        if item[0] == '.':
            sentences.append(sentence); sentence = []
    return sentences

def create_crf():
    #path to json files with spans
    path = '/Users/romapatel/Documents/PICO-data/seq-models/data/'
    #path to text files with documents
    docpath = '/Users/romapatel/Desktop/PICO-data/docs/'
    #choose aggregation from (hmm, mv, union)
    aggr = 'hmm'
    f = open(path + 'PICO-annos-crowd-agg.json', 'r')

    crowd, gold = {}, {}
    for line in f:
        temp = json.loads(line)
        crowd[temp['docid']] = temp

    f = open(path + 'PICO-annos-gold-agg.json', 'r')
    for line in f:
        temp = json.loads(line)
        gold[temp['docid']] = temp

    crf_crowd, crf_gold = {}, {}
    
    for docid in crowd:
        f = open(docpath + docid + '.txt', 'r')
        text = ''
        for line in f: text += line
        tokens = tokenize(text);
        tokens = [item for item in tokens if len(item[0]) > 0]
        indices = [[item[1], item[2]] for item in tokens]; tags = [[0, 0, 0] for item in indices]

        for span in crowd[docid]['Participants'][aggr]:
            for i in range(len(indices)):
                if indices[i][0] >= span[0] and indices[i][1] <= span[1]:
                    tags[i][0] = 1

        
        for span in crowd[docid]['Intervention'][aggr]:
            for i in range(len(indices)):
                if indices[i][0] >= span[0] and indices[i][1] <= span[1]:
                    tags[i][1] = 1

        for span in crowd[docid]['Outcome'][aggr]:
            for i in range(len(indices)):
                if indices[i][0] >= span[0] and indices[i][1] <= span[1]:
                    tags[i][2] = 1

        pos_tags= pos_tag([item[0] for item in tokens])
        tuples = [[pos_tags[i][0], 0, pos_tags[i][1], tags[i][0], tags[i][1], tags[i][2]] for i in range(len(tags))]                   

        sentences = sentence_split(tuples)
        crf_crowd[docid] = sentences

        print docid

    f = open(path + 'crf_crowd.json', 'w+')
    f.write(json.dumps(crf_crowd))
    

    crowd = gold
    for docid in crowd:
        f = open(docpath + docid + '.txt', 'r')
        text = ''
        for line in f: text += line
        tokens = tokenize(text);
        tokens = [item for item in tokens if len(item[0]) > 0]
        indices = [[item[1], item[2]] for item in tokens]; tags = [[0, 0, 0] for item in indices]

        for span in crowd[docid]['Participants'][aggr]:
            for i in range(len(indices)):
                if indices[i][0] >= span[0] and indices[i][1] <= span[1]:
                    tags[i][0] = 1

        
        for span in crowd[docid]['Intervention'][aggr]:
            for i in range(len(indices)):
                if indices[i][0] >= span[0] and indices[i][1] <= span[1]:
                    tags[i][1] = 1

        for span in crowd[docid]['Outcome'][aggr]:
            for i in range(len(indices)):
                if indices[i][0] >= span[0] and indices[i][1] <= span[1]:
                    tags[i][2] = 1

        pos_tags= pos_tag([item[0] for item in tokens])
        tuples = [[pos_tags[i][0], 0, pos_tags[i][1], tags[i][0], tags[i][1], tags[i][2]] for i in range(len(tags))]                   

        sentences = sentence_split(tuples)
        crf_crowd[docid] = sentences

        print docid

    f = open(path + 'crf_gold.json', 'w+')
    f.write(json.dumps(crf_crowd))       

def create_lstm():
    
    #path to json files with spans
    path = '/Users/romapatel/Documents/PICO-data/seq-models/data/'
    #path to text files with documents
    docpath = '/Users/romapatel/Desktop/PICO-data/docs/'
    #choose aggregation from (hmm, mv, union)
    aggr = 'hmm'
    f = open(path + 'PICO-annos-crowd-agg.json', 'r')

    crowd, gold = {}, {}
    for line in f:
        temp = json.loads(line)
        crowd[temp['docid']] = temp

    f = open(path + 'PICO-annos-gold-agg.json', 'r')
    for line in f:
        temp = json.loads(line)
        gold[temp['docid']] = temp

    #specify train, test, dev docids
    train = [docid for docid in crowd.keys() if docid not in gold.keys()]
    gold = [docid for docid in gold.keys()]

    fw = open(path + 'train.txt', 'w+')
    for docid in train:
        print docid
        fw.write('-DOCSTART- -X- O O\n\n')
        f = open(docpath + docid + '.txt', 'r')
        text = ''
        for line in f: text += line
        tokens = tokenize(text);
        tokens = [item for item in tokens if len(item[0]) > 0]
        indices = [[item[1], item[2]] for item in tokens]; tags = [[0, 0, 0] for item in indices]

        for span in crowd[docid]['Participants'][aggr]:
            for i in range(len(indices)):
                if indices[i][0] >= span[0] and indices[i][1] <= span[1]:
                    tags[i][0] = 1

        
        for span in crowd[docid]['Intervention'][aggr]:
            for i in range(len(indices)):
                if indices[i][0] >= span[0] and indices[i][1] <= span[1]:
                    tags[i][1] = 1

        for span in crowd[docid]['Outcome'][aggr]:
            for i in range(len(indices)):
                if indices[i][0] >= span[0] and indices[i][1] <= span[1]:
                    tags[i][2] = 1

        pos_tags= pos_tag([item[0] for item in tokens])
        tuples = [[pos_tags[i][0], 0, pos_tags[i][1], tags[i][0], tags[i][1], tags[i][2]] for i in range(len(tags))]                   

        sentences = sentence_split(tuples)
        for sentence in sentences:
            for token in sentence:
                if token[3] == 1: s = token[0] + ' ' + token[2] + ' ' + 'P' + '\n'
                elif token[4] == 1: s = token[0] + ' ' + token[2] + ' ' + 'I' + '\n'
                elif token[5] == 1: s = token[0] + ' ' + token[2] + ' ' + 'O' + '\n'
                else: s = token[0] + ' ' + token[2] + ' ' + 'N' + '\n'
                fw.write(s)
                fw.write('\n')
                
    
    crowd = gold
    fw = open(path + 'gold.txt', 'w+')
    for docid in gold:
        fw.write('-DOCSTART- -X- O O\n\n')
        f = open(docpath + docid + '.txt', 'r')
        text = ''
        for line in f: text += line
        tokens = tokenize(text);
        tokens = [item for item in tokens if len(item[0]) > 0]
        indices = [[item[1], item[2]] for item in tokens]; tags = [[0, 0, 0] for item in indices]

        for span in crowd[docid]['Participants'][aggr]:
            for i in range(len(indices)):
                if indices[i][0] >= span[0] and indices[i][1] <= span[1]:
                    tags[i][0] = 1

        
        for span in crowd[docid]['Intervention'][aggr]:
            for i in range(len(indices)):
                if indices[i][0] >= span[0] and indices[i][1] <= span[1]:
                    tags[i][1] = 1

        for span in crowd[docid]['Outcome'][aggr]:
            for i in range(len(indices)):
                if indices[i][0] >= span[0] and indices[i][1] <= span[1]:
                    tags[i][2] = 1

        pos_tags= pos_tag([item[0] for item in tokens])
        tuples = [[pos_tags[i][0], 0, pos_tags[i][1], tags[i][0], tags[i][1], tags[i][2]] for i in range(len(tags))]                   

        sentences = sentence_split(tuples)
        for sentence in sentences:
            for token in sentence:
                if token[3] == 1: s = token[0] + ' ' + token[2] + ' ' + 'P' + '\n'
                elif token[4] == 1: s = token[0] + ' ' + token[2] + ' ' + 'I' + '\n'
                elif token[5] == 1: s = token[0] + ' ' + token[2] + ' ' + 'O' + '\n'
                else: s = token[0] + ' ' + token[2] + ' ' + 'N' + '\n'
                fw.write(s)
                fw.write('\n')

        

    
if __name__ == "__main__":
    #create_crf()
    create_lstm()

