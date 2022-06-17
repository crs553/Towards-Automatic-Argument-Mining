import numpy as np

from dataReader.dataReader import Reader, get_train_test_split
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from itertools import chain
import pandas as pd
import string
import joblib
from os import getcwd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

from sklearn import svm
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer


class SVM_argClassification():
    def __init__(self, path, vec = None, clf = None):
        """
        Support Vector Machine initialisation
        :param path:
        """
        self.datareader = Reader(path)

        self.path = path
        self.embed_model = None
        self.vectoriser_mod = False if vec is None else vec
        self.clf = clf if clf is not None else make_pipeline(StandardScaler(with_mean=False), svm.SVC(gamma='auto'))

        self.x_train, self.x_test, self.y_train, self.y_test = self.prep_data_arg()
        # self.target = (0, 1, 2)  # 0 is not a major claim, 1 is a major claim
        self.is_trained = False

    def prep_data_arg(self):
        """PReps the training a test data"""
        data = self.datareader.load_from_directory()
        print("Directory loaded")

        # get all textual data
        text_data = data.fullText1.unique()

        # split textual data into sentences
        all_sentences = [sent_tokenize(x.lower()) for x in text_data]
        flatten = list(chain(*all_sentences))

        originals = data['originalArg1'].to_list()
        origin_temp = data['originalArg2'].to_list()
        originals.extend(origin_temp)
        originals = [x.lower() for x in originals]
        originals = set(originals)

        # position of each sentence in relation to its document
        pos_sents = []
        for fullt in all_sentences:
            len_full = len(fullt)
            for i, sent in enumerate(fullt):
                pos_sents.append(i / len_full)

        # get each sentence as a flattened array and if it is argumentative or not
        sent_feature, y_vals = [], []
        for x in flatten:
            temp = 0
            for i, y in enumerate(originals):
                if x in y or x == y:
                    temp = 1  # 1 means argumentative
                    break
            y_vals.append(temp)
            sent_feature.append(x)

        # get text position in fulltext as float
        pos_features = []
        for sents in all_sentences:
            len_text = len(sents)
            for i, sent in enumerate(sents):
                for curr_sent in flatten:
                    if curr_sent == sent:
                        pos_features.append(i / len_text)
                        break

        # punctuation number
        num_puncts_lst = []
        for sents in all_sentences:
            for sent in sents:
                val = sum([1 for char in sent[:-1] if char in string.punctuation])
                num_puncts_lst.append(val)

        # three words before

        punct_rem_tokeniser = RegexpTokenizer(r'\w+')
        sent_unpunct = []
        [sent_unpunct.append([punct_rem_tokeniser.tokenize(x) for x in texts]) for texts in all_sentences]

        all_before = []
        all_after = []
        for text in sent_unpunct:
            for i in range(len(text)):
                before = '$$$'
                after = '$$$'
                if i != 0:
                    before = ' '.join(text[i-1][-3:])
                if i<(len(text)-1):
                    after = ' '.join(text[i+1][:3])
                all_before.append(before)
                all_after.append(after)


        # raise ValueError

        # create feature dataframe
        df = {'sent_vector': sent_feature, 'pos_feature': pos_features, 'puct_num': num_puncts_lst,
              'before': all_before, 'after': all_after}
        df = pd.DataFrame(df)
        print("Features created")

        if self.vectoriser_mod is False:
            self.vectoriser_mod = TfidfVectorizer(stop_words="english", max_features=5000, decode_error="ignore",
                                          ngram_range=(1, 2))

        mapper = DataFrameMapper([('sent_vector', self.vectoriser_mod),
                                  ('before', self.vectoriser_mod),
                                  ('after', self.vectoriser_mod),
                                  (['pos_feature', 'puct_num'], None)])

        features = mapper.fit_transform(df)

        x_train, x_test, y_train, y_test = train_test_split(features, y_vals)

        return x_train, x_test, y_train, y_test

    def train_arg(self):
        self.clf.fit(self.x_train, self.y_train)

        print("Training Complete")

    def test_arg(self):
        return self.clf.predict(self.x_test)

    def score(self, y_pred):
        print(accuracy_score(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred))


def run(saveTloadF = True):
    path = getcwd()
    path += "/ArgumentAnnotatedEssays-2.0/"

    vec = None
    clf = None
    if not saveTloadF:
        name = input("Enter Model name with file extension")
        with open(name, 'rb') as r:
            vec, clf = joblib.load(r)

    ml_model = SVM_argClassification(path,vec, clf)

    print("Training model")
    if saveTloadF:
        ml_model.train_arg()

    print("Predicting")
    y_pred = ml_model.test_arg()

    print("Scoring")
    ml_model.score(y_pred)
    if saveTloadF:
        name = input("Enter model save_name:")
        ml_model.vectoriser_mod.stop_words_ = None
        with open(f'{name}.arg', 'wb') as r:
            joblib.dump((ml_model.vectoriser_mod, ml_model.clf), r)

    x_train = ml_model.x_train
    y_train = ml_model.y_train
    x_test = ml_model.x_test
    y_test = ml_model.y_test
    vector = ml_model.vectoriser_mod

    return x_train, y_train, x_test, y_test, y_pred
