import xml.etree.ElementTree as ET
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import fuzzywuzzy as fuzz

non_sw = set(('more', 'most', 'no', 'non' 'not', 'only', 'such', 'few', 'so', 'too', 'very', 'just', 'once'))
stopWords = set(stopwords.words('english')) - non_sw
sw = ['anything', 'everything', 'thing', 'anybody', 'hey', 'anyone']
for w in sw:
    stopWords.add(w)


def parseXML(file):

    # create element tree object
    tree = ET.parse(file)
    root = tree.getroot()

    reviews = defaultdict(dict)
    for child in root.iter():

        # Setting sentence id as dictionary key
        if child.tag == 'sentence':
            id = child.attrib['id']
            reviews[id] = defaultdict()

        # Adding text as a value
        elif child.tag == 'text':
            text = child.text
            reviews[id]['text'] = text

    return reviews


def parseXML(file):

    # create element tree object
    tree = ET.parse(file)
    root = tree.getroot()

    reviews = defaultdict(dict)
    for child in root.iter():

        # Setting sentence id as dictionary key
        if child.tag == 'sentence':
            id = child.attrib['id']
            reviews[id] = defaultdict()

        # Adding text as a value
        elif child.tag == 'text':
            text = child.text
            reviews[id]['text'] = text

        # Creating a list for opinions (if any)
        elif child.tag == 'Opinions':
            reviews[id]['opinions'] = []

        # Populating lists for opinions
        elif child.tag == 'Opinion':
            reviews[id]['opinions'].append(child.attrib)

    return reviews


def filter_data(reviews):
    # getting rid of sentences with no labels
    for id, doc in list(reviews.items()):
        # doc has no opinions
        if len(doc) == 1:
            del reviews[id]
    return reviews


def flatten_attributes(reviews):
    # getting rid of sentences with no labels
    for id, rev in list(reviews.items()):
        opinions = rev['opinions']
        for op in opinions:
            split = op['category'].split("#")

            # for LAPTOP get attributes
            if split[0] == 'LAPTOP':
                op['category'] = split[1]

            # other entities stay the same
            else:
                op['category'] = split[0]
    return reviews


def tokenize(reviews):

    for id in reviews:
        text = reviews[id]['text'].lower()
        reviews[id]['text'] = word_tokenize(text)

    return reviews


def clean_text(reviews):

    for id in reviews:
        # all to lower case
        text = reviews[id]['text'].lower()
        # getting rid of contractions and special chars
        reviews[id]['text'] = reviews[id]['text'].replace("n't", " not")
        reviews[id]['text'] = reviews[id]['text'].replace("'s", " is")
        reviews[id]['text'] = reviews[id]['text'].replace("'m", " am")
        reviews[id]['text'] = reviews[id]['text'].replace("'ve", " have")
        reviews[id]['text'] = reviews[id]['text'].replace('``', "")
        reviews[id]['text'] = reviews[id]['text'].replace('w/', "with")
        reviews[id]['text'] = reviews[id]['text'].replace('&apos;', "")
        reviews[id]['text'] = reviews[id]['text'].replace('/', " or ")
        reviews[id]['text'] = reviews[id]['text'].replace('-', " ")
        reviews[id]['text'] = reviews[id]['text'].replace("'n", "n")
        reviews[id]['text'] = reviews[id]['text'].replace("'c", "c")

        if text.endswith('..') and text != '...':
            reviews[id]['text'] = reviews[id]['text'].replace("..", "")
        if text.endswith('-'):
            reviews[id]['text'] = reviews[id]['text'].replace("-", "")

        # simplifying repetition of characters
        match = re.finditer(r"([a-zA-Z])\1{3,}", text, re.IGNORECASE)
        reps = [m.group(0) for m in match]
        if len(reps) > 0:
            for j in range(len(reps)):
                reviews[id]['text'] = reviews[id]['text'].replace(reps[j], reps[j][0])

    return reviews


def fuzzy_match(words, assoc_dict, assoc_nouns):
    # method that matches similar words to already collected words.

    for word in words:

        max_ratio = 0
        max_cat = ''
        for cat, tokens in assoc_dict.items():
            for token in tokens:
                ratio = fuzz.ratio(word, token)
                if ratio > max_ratio:
                    max_ratio = ratio
                    max_cat = cat

        if max_cat != '' and max_ratio > 84:
            assoc_dict[max_cat].append(word)
            assoc_nouns.append(word)

    return assoc_dict, assoc_nouns