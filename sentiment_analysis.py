import pandas

import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import string
from os import path
from random import shuffle


def filter_list(sentence):
    sentence_blob = TextBlob(sentence)
    tokens = [word for word in sentence_blob.words if word != 'user']
    filtered_tokens = [token for token in tokens if re.match(r'[^\W\d]*$', token)]
    clean_tokens = [word for word in filtered_tokens if word.lower() not in stopwords.words('english')]

    return clean_tokens


def normalize_sentence(sentence_list):
    lem = WordNetLemmatizer()
    normalized_sentence = [lem.lemmatize(word, 'v') for word in sentence_list]

    return normalized_sentence


def process_data(sentence):
    sentence_list = filter_list(sentence)

    return normalize_sentence(sentence_list)


def process_file(file_name, label, lines):
    with open(file_name, 'r') as fd:
        for line in fd:
            line = process_data(line)
            line = ' '.join(line)
            line += ',' + str(label)
            lines.append(line)


def load_movie_data():
    if not path.exists("data/movie.csv"):
        lines = []

        process_file('./data/polarity-pos.txt', 1, lines)
        process_file('./data/polarity-neg.txt', 0, lines)

        shuffle(lines)

        with open("./data/movie.csv", "w+") as out:
            out.write("tweet,label\n")
            for line in lines:
                out.write(line + "\n")

    return pandas.read_csv("data/movie.csv")


def test_model(input_data):
    tweets = input_data['tweet']
    labels = input_data['label']
    fixed_data_tweets = tweets[pandas.notnull(tweets)]
    fixed_data_labels = labels[pandas.notnull(tweets)]

    sentence_train, sentence_test, label_train, label_test = train_test_split(
        fixed_data_tweets, fixed_data_labels, test_size=0.25)

    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB()),
    ])

    pipeline.fit(sentence_train, label_train)
    predictions = pipeline.predict(sentence_test)
    print(classification_report(label_test, predictions, digits=4))


movie_data = load_movie_data()

test_model(movie_data)
