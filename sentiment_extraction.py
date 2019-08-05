from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
import io
lem = WordNetLemmatizer()

# Sentiment Shifters
negation = ['no',
            'not',
            'without',
            'never',
            'none',
            'nobody',
            'nothing',
            'neither',
            'nowhere',
            'hardly',
            'barely',
            'rarely',
            'scarcely']

contrast = ['but',
            'however',
            'unfortunately',
            'though',
            'although',
            'despite',
            'yet',
            'nonetheless',
            'nevertheless',
            'still',
            'except']


def get_lexicon(file):
    lines = io.open(file).readlines()
    lexicon = [line.strip() for line in lines if line.startswith(';') is False]

    return lexicon


def get_verbal_shifters(file):

    lines = io.open(file).readlines()
    lines = [line.strip() for line in lines]
    shifters = [line.split(',') for line in lines]
    shifters = [verb[0] for verb in shifters if verb[1] == 'shifter']

    return shifters


def score_function(id, review, lemma_dict, feature, adj_dict, positive_lexicon, negative_lexicon, neg_words, verbal_shifters):
    feat_score = 0
    for adj in adj_dict[id]['text']:
        adj = lem.lemmatize(adj)
        text = review['text']
        w_sent_score = sentiment_score(text, adj, lemma_dict, positive_lexicon, negative_lexicon, neg_words, verbal_shifters)
        dist = opinion_feat_dist(review['text'], lemma_dict, feature, adj)
        if dist != 0:
            feat_score += (w_sent_score/dist)

    return feat_score


def sentiment_score(text, adj, lemma_dict, positive_lexicon, negative_lexicon, neg_words, verbal_shifters):

    adj_lem = adj
    lemmas_adj = lemma_dict[adj]
    for lem_adj in lemmas_adj:
        if lem_adj in text:
            adj = lem_adj

    adj_ind = text.index(adj)
    prev_window = text[(adj_ind-2):adj_ind]

    # checking for shifters
    shift = False
    for token in prev_window:
        if token in neg_words or token in verbal_shifters:
            shift = True

    if len(prev_window) > 0:
        if prev_window[0] in neg_words or verbal_shifters:
            if prev_window[1] in neg_words or verbal_shifters:
                shift = False

    sent_score = 0
    if shift is False:
        if adj_lem in positive_lexicon:
            sent_score = 1
        elif adj_lem in negative_lexicon:
            sent_score = -1

    else:
        if adj_lem in positive_lexicon:
            sent_score = -1
        elif adj_lem in negative_lexicon:
            sent_score = 1

    # BUT-HANDLING: when no sentiment was marked
    if sent_score == 0:
        but_ind = -1
        for word in range(len(text)):
            if text[word] in contrast:
                but_ind = word

        # if adj is after but clause
        clause_score = 0
        if adj_ind < but_ind and but_ind > 0:
            text = text[but_ind+1:]
            for word in range(len(text)):
                clause_score += sentiment_score(text, text[word], lemma_dict, positive_lexicon, negative_lexicon, neg_words, verbal_shifters)
            sent_score = clause_score*(-1)

        elif adj_ind > but_ind and but_ind > 0:
            text = text[:but_ind]
            for word in range(len(text)):
                clause_score += sentiment_score(text, text[word], lemma_dict, positive_lexicon, negative_lexicon, neg_words, verbal_shifters)
            sent_score = clause_score*(-1)

    return sent_score


def opinion_feat_dist(text, lemma_dict, feature, adj):
    '''Distance between an opinion and a feature in a given text'''

    lemmas = lemma_dict[feature]
    for lem in lemmas:
        if lem in text:
            feature = lem

    lemmas_adj = lemma_dict[adj]
    for lem_adj in lemmas_adj:
        if lem_adj in text:
            adj = lem_adj

    dist = text.index(feature) - text.index(adj)

    return abs(dist)


def get_lemma_dict(reviews):

    lemma_dict = defaultdict(list)

    for id, rev in reviews.items():
        for token in rev['text']:
            lemma = lem.lemmatize(token)
            lemma_dict[lemma].append(token)

    for lemma, tokens in lemma_dict.items():
        lemma_dict[lemma] = list(set(tokens))

    return lemma_dict


def sent_extraction(reviews, lemma_dict, adj_dict, pos_lex, neg_lex, neg_words, verbal_shifters):

    for id, rev in reviews.items():
        for op in rev['opinions']:
            if 'feature' in op.keys():
                feature = op['feature']
                score = score_function(id, reviews[id], lemma_dict, feature, adj_dict, pos_lex, neg_lex, neg_words, verbal_shifters)
                op['score'] = score
                if score > 0:
                    op['sentiment'] = 'positive'
                elif score < 0:
                    op['sentiment'] = 'negative'
                else:
                    op['sentiment'] = 'neutral'
    return reviews

verbal_shifters = get_verbal_shifters('verbal_shifters.csv')
