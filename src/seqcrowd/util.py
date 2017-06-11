class instance:
    """
    an instance
    """

    def __init__(self, features, label, word = None):
        self.features = features
        self.label = label
        if word != None:
            self.word = word

class crowdlab:
    """
    a sentence labeled by crowd
    """

    def __init__(self, wid, sid, sen):
        """

        :param sen: list of labels
        :param wid: worker id
        :param sid: sentence id
        """
        self.sen = sen
        self.wid = wid
        self.sid = sid


class crowd_data:
    """
    a dataset with crowd labels
    """

    def __init__(self, sentences, crowdlabs):
        """

        :param sentences: list of sen
        :param crowdlabs: list of list of crowdlab
        each "list of crowdlab" is a list of crowd labels for a particular sentence
        """
        self.sentences = sentences
        self.crowdlabs = crowdlabs

    def get_labs(self, i, j):
        """
        return all labels for sentence i, position j
        """
        res = []
        for c in self.crowdlabs[i]:
            res.append(c.sen[j])
        return res

    def get_lw(self, i, j):
        """
        return all labels/wid for sentence i, position j
        """
        res = []
        for c in self.crowdlabs[i]:
            res.append( (c.sen[j], c.wid))
        return res



def get_features(word, prev, next):
    word = word.strip()
    a = []
    if len(word) > 0:
        if word[0].isupper():
            a.append("*U")
        if word[0].isdigit():
            a.append("*D")
        return a

    if len(word) > 0:
        if len(word) > 1:
            a.append(word[:2])
            a.append(word[-2:])
            if len(word) > 2:
                a.append(word[:3])
                a.append(word[-3:])

    return a


def process_word(word):
    """
    to standardize word to put in list of features
    :param word:
    :return:
    """
    # return word.lower()
    return word


def get_first_word(s):
    x = s.strip().split()
    if len(x) < 1:
        return ""
    return process_word(x[0])


def get_prev_next(input, i):
    n = len(input)
    if i == 0:
        return ("*ST", get_first_word(input[i + 1]))
    if i == n - 1:
        return (get_first_word(input[i - 1]), "*EN")
    return (get_first_word(input[i - 1]), get_first_word(input[i + 1]))


def build_index(input):

    list_labels = []
    list_features = []

    for i, line in enumerate(input):
        a = line.strip().split()
        #print a
        if a == []:
            continue
        # last word is label
        list_labels.append(a[-1])
        for f in a[:-1]:
            list_features.append(process_word(f))

        prev, next = get_prev_next(input, i)
        list_features.extend(get_features(a[0], prev, next))

    # get unique labels and features
    list_labels = sorted(list(set(list_labels)))
    list_features = sorted(list(set(list_features)))

    # maps from word to labels/features
    labels = {}
    features = {}
    for i, l in enumerate(list_labels):
        labels[l] = i + 1  # label 0 = unseen label
    for i, f in enumerate(list_features):
        features[f] = i + 1  # features 0 = OOV

    return features, labels


def extract(input, features, labels, keep_word=False):
    sentences = []
    sentence = []
    for i, line in enumerate(input):
        a = line.strip().split()
        if a == []:
            if sentence != []:
                sentences.append(sentence)
                sentence = []
            continue
        list_f = []

        prev, next = get_prev_next(input, i)

        if process_word(a[0]) in features:
            list_f.append(features[process_word(a[0])])

            for f in get_features(a[0], prev, next):
                if f in features:
                    list_f.append(features[f])
        else:
            for f in get_features(a[0], prev, next):
                if f in features:
                    list_f.append(features[f])

        if list_f == []:
            list_f = [0]

        label = labels[a[-1]] if a[-1] in labels else 0
        if keep_word:
            i = instance(list_f, label, a[0])
        else:
            i = instance(list_f, label)
        sentence.append(i)
    return sentences


def get_obs(sentence):
    """
    get the seq of observations, each observation = list of features
    :param sentence:
    :return:
    """
    res = []
    for i in sentence:
        res.append(i.features)
    return res


def make_sen(sen, res):
    for s, r, in zip(sen, res):
        for i, x in zip(s,r):
            i.label = x
