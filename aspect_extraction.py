from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

lem = WordNetLemmatizer()


def get_comatrix(reviews, vocabulary, entities):

    # initialization of matrix
    matrix = np.zeros((len(vocabulary), len(entities)))

    # keys for the matrix mapping words to indices
    word_keys = {}
    i = 0
    for w in vocabulary:
        lemma = lem.lemmatize(w)
        word_keys[lemma] = i
        i += 1

    cat_keys = {}
    i = 0
    for e in entities:
        cat_keys[e] = i
        i += 1

    # populating the matrix
    for id,rev in reviews.items():
        tokens = rev['text']
        categories = [opinion['category'] for opinion in rev['opinions']]
        for tok in tokens:
            for cat in categories:
                i = lem.lemmatize(tok)
                j = cat
                if i in word_keys.keys():
                    index_i = word_keys[i]
                    index_j = cat_keys[j]
                    matrix[index_i, index_j] += 1

    # multiplying the matrix by 10 -- smoothing
    matrix += 1

    return matrix, word_keys, cat_keys


def compute_ppmi(reviews, vocabulary, categories):

    # PMI(w,c) = max (log P(w,c)/P(w)P(c) ,0)
    # p(w,c) = count(w,c)/count(w)
    # p(w) = count(w)/total
    # p(c) = count(c)/total

    matrix, word_keys, cat_keys = get_comatrix(reviews, vocabulary, categories)
    prob_word_cat = defaultdict(float)

    #total nr of occurences
    sum = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            sum += matrix[i, j]

    ppmi_matrix = np.zeros(matrix.shape)

    for word in vocabulary:
        for cat in categories:
            lemma = lem.lemmatize(word)

            index_w = word_keys[lemma]
            index_c = cat_keys[cat]
            prob_word_cat[index_w, index_c] = matrix[index_w, index_c] / sum
            word_prob = np.sum(matrix, axis=1)[index_w] / sum
            cat_prob = np.sum(matrix, axis=0)[index_c] / sum
            ppmi_matrix[index_w, index_c] = prob_word_cat[index_w, index_c] / (word_prob * cat_prob)

    return ppmi_matrix, word_keys, cat_keys


def get_id_map(dataset):

    # keys for the matrix mapping words to indices
    id_map = {}
    i = 0
    for id in dataset:
        id_map[i] = id
        i += 1
    return id_map


def get_X(train, id_map, features, categories, cat_dict):

    # vocab occurrence matrix
    matrix = np.zeros((len(train), len(features)))
    for i in id_map:
        id = id_map[i]
        #print(type(id))
        for j in range(len(features)):
            text = train[id]['text']
            for token in text:
                if token in features[j]:
                    matrix[i][j] = 1
    return matrix


def get_Y(train, id_map, category):

    # label matrix
    matrix = np.zeros(len(train),)

    for i in id_map:
        id = id_map[i]
        opinions = train[id]['opinions']
        for opinion in opinions:
            cat = opinion['category']
            if category == cat:
                matrix[i] = 1

    return matrix


def train_aspect_classifiers(train, id_map, categories, feat_dict, solver):

    # training classifier
    avg_scores = defaultdict(dict)
    classifiers = defaultdict(dict)
    for cat in categories:
        features = feat_dict[cat]
        x = get_X(train, id_map, features, categories, feat_dict)
        y = get_Y(train, id_map, cat)
        kf = model_selection.KFold(n_splits=5, shuffle=False)
        classifier = LogisticRegression(random_state=0, solver=solver, max_iter=100).fit(x, y)
        classifiers[cat]['model'] = classifier
        score = model_selection.cross_val_score(classifier, x, y, cv=kf)
        classifiers[cat]['score'] = score

        scoring = ['accuracy', 'precision_macro', 'recall_macro']
        metrics = ['test_accuracy', 'test_precision_macro', 'test_recall_macro']
        scores = model_selection.cross_validate(classifier, x, y, scoring=scoring, cv=5, return_train_score = False)

        for met in metrics:
            avg_scores[cat][met] = scores[met].mean()

    return classifiers


def predict_aspect(reviews, id_map, categories, vocab, classifiers):
    '''Predicts category for each review and adds it to the dictionary'''

    preds = defaultdict(list)
    for cat, model in classifiers.items():
        features = vocab[cat]
        revs = get_X(reviews, id_map, features, categories, vocab)
        y_pred = model['model'].predict(revs)
        preds[cat] = y_pred

    for cat, y_pred in preds.items():
        for i in range(len(y_pred)):
            id = id_map[i]

            # checking if list in key already exists
            try:
                tuples = reviews[id]['absa']
            except KeyError:
                reviews[id]['absa'] = []

            # if predicted, add category to list
            if y_pred[i] == 1:
                reviews[id]['absa'].append({'category': cat})

    return reviews


def feature_liaison(reviews, noun_dict, adj_dict, ppmi_matrix, word_keys, cat_keys):
    for id, review in noun_dict.items():

        opinions = reviews[id]['opinions']
        features = review['text']
        absa_tuples = []
        cat_feat_ppmi = defaultdict(dict)
        for feature in list(features):
            feature = lem.lemmatize(feature)
            cat_list = []
            for op in opinions:
                cat = list(op.values())[0]
                cat_list.append(cat)    # collecting cats from predictions
                cat_ind = cat_keys[cat]
                if feature in word_keys.keys():
                    feat_ind = word_keys[feature]
                    cat_feat = ppmi_matrix[feat_ind][cat_ind]
                    cat_feat_ppmi[cat][feature] = cat_feat

            for cat in cat_list:
                max_word = max(cat_feat_ppmi[cat], key=lambda k: cat_feat_ppmi[cat][k])

                for op in opinions:
                    if op['category'] == cat:
                        op['feature'] = max_word

    return reviews
