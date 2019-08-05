import preprocessing
import aspect_extraction
import sentiment_extraction
from nltk.parse import CoreNLPParser
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
import copy
import pickle


# Core NLP Server services
parser = CoreNLPParser(url='http://localhost:9001')
core_tagger = CoreNLPParser(url='http://localhost:9001', tagtype='pos')
lem = WordNetLemmatizer()


def retreive_tags(train):

    entities = []

    for doc in train.values():
        labels = doc['opinions']

        for opinion in labels:
            entity = opinion['category']
            entities.append(entity)

    entities = list(set(entities))

    return entities


def pos_tagger(text, tagger):
    # using the CoreNLP server
    return list(tagger.tag(text))


def parse_text(text, parser):
    return list(parser.parse(text))


def collect_pos(reviews, stopWords):
    # collecting aspects from nouns
    noun_dict = copy.deepcopy(reviews)
    adj_dict = copy.deepcopy(reviews)
    pos_dict = copy.deepcopy(reviews)

    for id, rev in noun_dict.items():
        text = rev['text']
        pos_sent = pos_tagger(text, core_tagger)
        all = []
        nouns = []
        adjs = []
        for pos in pos_sent:
            token = pos[0]

            if 'NN' in pos[1] and token not in stopWords:
                nouns.append(token)
                all.append(token)

            elif 'JJ' in pos[1]and token not in stopWords:
                adjs.append(token)
                all.append(token)

        noun_dict[id]['text'] = nouns
        adj_dict[id]['text'] = adjs
        pos_dict[id]['text'] = all

    return noun_dict, adj_dict, pos_dict


def get_vocab(reviews):
    vocab = []
    for id, rev in reviews.items():
        vocab.extend(rev['text'])
    return list(set(vocab))


def evaluate_polarity(reviews, categories):
    evaluation = defaultdict(dict)
    tp = 0
    pre_total = 0
    pre_tp = 0
    rec_total = 0
    rec_tp = 0
    total = 0

    for cat in categories:
        for id, rev in reviews.items():
            for op in rev['opinions']:
                if 'sentiment' in op.keys():
                    if op['polarity'] == op['sentiment']:
                        tp += 1
                    if cat == op['sentiment']:
                        pre_total += 1
                        if op['polarity'] == op['sentiment']:
                            pre_tp += 1
                    if cat == op['polarity']:
                        rec_total += 1
                        if op['polarity'] == op['sentiment']:
                            rec_tp += 1
                total += 1

        accu = (tp/total)
        prec = (pre_tp/pre_total)
        rec = (rec_tp/rec_total)
        evaluation[cat] = {'accuracy': accu, 'precision': prec, 'recall': rec}

    return evaluation


if __name__ == "__main__":

    print('Preprocessing...')

    train = preprocessing.parseXML('ABSA16_Laptops_Train_SB1_v2.xml')
    train = preprocessing.filter_data(train)
    train = preprocessing.flatten_attributes(train)
    train = preprocessing.clean_text(train)
    train = preprocessing.tokenize(train)

    test = preprocessing.parseXML('test_2016.xml')
    test = preprocessing.filter_data(test)
    test = preprocessing.flatten_attributes(test)
    test = preprocessing.clean_text(test)
    test = preprocessing.tokenize(test)
    train = {**train, **test} # merging train and test set

    # Aspect Extraction
    print('\nAspect Detection')
    categories = retreive_tags(train)
    noun_dict, adj_dict, pos_dict = collect_pos(train, preprocessing.stopWords)
    noun_vocab = get_vocab(noun_dict)
    adj_vocab = get_vocab(adj_dict)

    vocab = get_vocab(train)
    all_vocab = defaultdict(list)
    for cat in categories:
        all_vocab[cat] = vocab
    id_map = aspect_extraction.get_id_map(train)


    print('Computing PPMI...')
    ppmi_matrix, word_keys, cat_keys = aspect_extraction.compute_ppmi(train, vocab, categories)

    print('Training...')
    solvers = ['newton-cg', 'liblinear', 'sag']
    #classifiers = aspect_extraction.train_aspect_classifiers(train, id_map, categories, all_vocab, solvers[0])
    #pick = pickle.dump(classifiers, open('classifiers.pkl', 'wb'))
    classifiers = pickle.load(open('classifiers.pkl', 'rb'))


    print('\nAspect Extraction')
    # associating predicted categories with features (nouns)
    train = aspect_extraction.feature_liaison(train, noun_dict, adj_dict, ppmi_matrix, word_keys, cat_keys)


    print('\nSentiment extraction')
    # Sentiment Extraction
    polarities = ['positive', 'negative', 'neutral']
    pos_lex = './opinion-lexicon/positive-words.txt'
    neg_lex = './opinion-lexicon/negative-words.txt'

    pos_lex = sentiment_extraction.get_lexicon(pos_lex)
    neg_lex = sentiment_extraction.get_lexicon(neg_lex)

    op_lex = {'positive': pos_lex, 'negative': neg_lex}
    lemma_dict = sentiment_extraction.get_lemma_dict(train)

    neg_words = sentiment_extraction.negation
    verbal_shifters = sentiment_extraction.verbal_shifters

    train = sentiment_extraction.sent_extraction(train, lemma_dict, adj_dict, pos_lex, neg_lex, neg_words, verbal_shifters)

    print(evaluate_polarity(train, polarities))
