import readability

"""
OrderedDict([(u'Kincaid', 13.718888888888891),
                           (u'ARI', 16.876666666666672),
                           (u'Coleman-Liau', 19.141197777777776),
                           (u'FleschReadingEase', 28.765000000000025),
                           (u'GunningFogIndex', 18.311111111111114),
                           (u'LIX', 68.0),
                           (u'SMOGIndex', 15.24744871391589),
                           (u'RIX', 9.0),
                           (u'DaleChallIndex', 15.055966666666665)])),
             (u'sentence info',
              OrderedDict([(u'characters_per_word', 6.222222222222222),
                           (u'syll_per_word', 1.8888888888888888),
                           (u'words_per_sentence', 18.0),
                           (u'sentences_per_paragraph', 1.0),
                           (u'type_token_ratio', 1.0),
                           (u'characters', 112),
                           (u'syllables', 34),
                           (u'words', 18),
                           (u'wordtypes', 18),
                           (u'sentences', 1),
                           (u'paragraphs', 1),
                           (u'long_words', 9),
                           (u'complex_words', 5),
                           (u'complex_words_dc', 12)])),
             (u'word usage',
              OrderedDict([(u'tobeverb', 1),
                           (u'auxverb', 0),
                           (u'conjunction', 0),
                           (u'pronoun', 0),
                           (u'preposition', 3),
                           (u'nominalization', 0)])),
             (u'sentence beginnings',
              OrderedDict([(u'pronoun', 0),
                           (u'interrogative', 0),
                           (u'article', 0),
                           (u'subordination', 0),
                           (u'conjunction', 0),
                           (u'preposition', 0)]))])
"""

def _readability_grads(score_dict):
    keys = ['Kincaid', 'ARI', 'Coleman-Liau', 'FleschReadingEase', 'GunningFogIndex', 'LIX', \
            'SMOGIndex', 'RIX', 'DaleChallIndex']

    values = [score_dict['readability grades'][k] for k in keys]
    return values


def _sentence_info(score_dict):
    keys = ['characters_per_word', 'syll_per_word', 'words_per_sentence', 'sentences_per_paragraph', \
            'type_token_ratio', 'characters', 'syllables', 'words', 'sentences', 'paragraphs', \
            'long_words', 'complex_words', 'complex_words_dc']

    values = [score_dict['sentence info'][k] for k in keys]
    return values


def _word_usage(score_dict):
    keys = ['tobeverb', 'auxverb', 'conjunction', 'pronoun', 'preposition', 'nominalization']

    values = [score_dict['word usage'][k] for k in keys]
    return values


def _sentence_beginnings(score_dict):
    keys = ['pronoun', 'interrogative', 'article', 'subordination', 'conjunction', 'preposition']

    values = [score_dict['sentence beginnings'][k] for k in keys]
    return values


def transform(texts):

    res = []
    for text in texts:
        text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
        score_dict = readability.getmeasures(text)

        scores = _readability_grads(score_dict)
        scores.extend(_sentence_info(score_dict))
        scores.extend(_word_usage(score_dict))
        scores.extend(_sentence_beginnings(score_dict))

        res.append(scores)

    return res


if __name__ == '__main__':
    print transform(u"BACKGROUND The aim of this study was the examination of efficacy and tolerability of an application - form of the new combination of Xylometazoline with Dexpanthenol ( Nasic ) versus Xylometazoline alone . ")
