import numpy as np

from dataReader.dataReader import Reader, get_train_test_split
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from itertools import chain
import itertools
import math
import pandas as pd
import string
import joblib
import csv
from tqdm import tqdm
from os import getcwd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

from sklearn import svm
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
# from sklearn.multioutput import MultiOutputClassifier
from nltk.stem import WordNetLemmatizer


class SVM_argClassification():
    def __init__(self, path, datareader=None, vec=None, clf=None, build=True):
        """
        Support Vector Machine initialisation
        :param path:
        """
        self.datareader = datareader
        self.build = build
        self.overall_data = {}
        self.all_sents = list()
        self.data = list()
        self.path = path
        self.embed_model = None
        self.vectoriser_mod = False if vec is None else vec
        self.clf = clf if clf is not None else make_pipeline(StandardScaler(with_mean=False), svm.SVC(gamma='auto'))

        self.text_data = list()
        self.x_train_loc = list()
        self.x_test_loc = list()
        self.x_train, self.x_test, self.y_train, self.y_test = self.prep_data_arg()
        # self.target = (0, 1, 2)  # 0 is not a major claim, 1 is a major claim
        self.is_trained = False

    def prep_data_arg(self):
        """PReps the training a test data"""

        data = self.datareader.load_from_directory()

        self.data = data
        print("Directory loaded")

        # get all textual data
        text_data = data.fullText1.unique()
        pairs = []

        # split textual data into sentences
        all_sentences = [sent_tokenize(x.lower()) for x in text_data]

        flatten = list(chain(*all_sentences))

        originals = data['originalArg1'].to_list()
        origin_temp = data['originalArg2'].to_list()
        originals.extend(origin_temp)
        originals = [x.lower() for x in originals]
        originals = set(originals)

        if self.build:
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
                for y in originals:
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

            # three words before and after
            punct_rem_tokeniser = RegexpTokenizer(r'\w+')
            sent_unpunct = []
            [sent_unpunct.append([punct_rem_tokeniser.tokenize(x) for x in texts]) for texts in all_sentences]
            all_before = []
            all_after = []
            for text in sent_unpunct:
                for i in range(len(text)):
                    before = '$$$'  # if no word before or after append triple dollar
                    after = '$$$'
                    if i != 0:
                        before = ' '.join(text[i - 1][-3:])
                    if i < (len(text) - 1):
                        after = ' '.join(text[i + 1][:3])
                    all_before.append(before)
                    all_after.append(after)

            df = {'sent_vector': sent_feature, 'pos_feature': pos_features, 'puct_num': num_puncts_lst,
                  'before': all_before, 'after': all_after, 'y_vals': y_vals}

            df = pd.DataFrame(df)
            df['overall_index'] = df.index
            self.overall_data = df
            self.overall_data.to_pickle('arg_class_data.pkl')
        else:
            df = pd.read_pickle('arg_class_data.pkl')
            self.overall_data = df

        if self.vectoriser_mod is False:
            self.vectoriser_mod = TfidfVectorizer(stop_words="english", max_features=50, decode_error="ignore",
                                                  ngram_range=(1, 2))

        mapper = DataFrameMapper([('sent_vector', self.vectoriser_mod),
                                  ('before', self.vectoriser_mod),
                                  ('after', self.vectoriser_mod),
                                  (['pos_feature', 'puct_num'], None)])

        features = mapper.fit_transform(df)
        y_vals = df['y_vals'].to_list()
        x_train, x_test, y_train, y_test = train_test_split(features, y_vals)
        print("Features Created")
        return x_train, x_test, y_train, y_test

    def train_arg(self):
        self.clf.fit(self.x_train, self.y_train)

        print("Training Complete")

    def test_arg(self):
        return self.clf.predict(self.x_test)

    def score(self, y_pred):
        print(accuracy_score(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred))


class SVM_Relationship():
    def __init__(self, data, build=True):

        self.data = data

        labels = self.data.label.unique()
        labels = [abs(x) for x in labels.tolist()]
        # labels.extend([3.0])
        labels = list(set(labels))
        print(labels)
        self.labelenc = LabelEncoder()
        self.labelenc.fit(labels)

        self.vectoriser_mod = TfidfVectorizer(stop_words="english", max_features=50, decode_error="ignore",
                                              ngram_range=(1, 2))

        self.build = build
        self.relations = None
        self.clf = make_pipeline(StandardScaler(with_mean=False), svm.SVC(gamma='auto'))

    def build_relations(self):
        print("building relations database for all values")

        fullTexts = self.data.fullText1.unique()

        pairs = []
        all_sentences = [sent_tokenize(x) for x in fullTexts]

        for i, unique in enumerate(fullTexts):
            pairs.append((self.data.loc[self.data['fullText1'] == unique, 'fileid'].iloc[0], all_sentences[i]))

        all_pos = [i for (i, _) in pairs]

        all_sent_fileid = [(i, self.clean_str(sent)) for i, sents in enumerate(all_sentences) for sent in sents]
        all_fileid = [i for (i, _) in all_sent_fileid]

        relations = self.data.loc[:, ['originalArg1', 'originalArg2', 'label', 'fileid']]
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1520)

        if self.build:
            senta = []
            pos_a, pos_b = [], []
            sentb = []
            temp_file_pos = []
            labels = []
            pbar = tqdm(total=len(all_sentences))
            for i, doc in enumerate(all_sentences):
                len_doc = len(doc)
                pos = [(j / len_doc, x) for j, x in enumerate(doc)]
                prod = [(x, x) for x in doc]
                for (sent1, sent2) in prod:
                    label_temp = self.data.loc[((self.data['originalArg1'] == sent1) &
                                                (self.data['originalArg2'] == sent2)), 'label'].tolist()
                    label_temp = [abs(x) for x in label_temp]
                    label_temp = list(set(label_temp))

                    if len(label_temp) == 0:
                        continue
                    labels.append(label_temp)
                    senta.append(sent1)
                    pos_a.append([a for (a, x) in pos if x == sent1][0])
                    sentb.append(sent2)
                    pos_b.append([a for (a, x) in pos if x == sent2][0])
                    temp_file_pos.append(all_pos[i])
                pbar.update(1)

            df = {"sent1": senta, "sent2": sentb, "fileid": temp_file_pos, "labels": labels,
                  'pos_a': pos_a, 'pos_b': pos_b}
            self.relations = pd.DataFrame(df)
            self.relations.to_pickle('relational_dataset.pkl')
        else:
            self.relations = pd.read_pickle('relational_dataset.pkl')
            print("Relational_dataset loaded")

    def get_features(self):
        mapper = DataFrameMapper([('sent1', self.vectoriser_mod),
                                  ('sent2', self.vectoriser_mod),
                                  (['pos_a', 'pos_b'], None)
                                  ])
        features = mapper.fit_transform(self.relations)
        return features

    def get_labels(self):
        labels = self.relations['labels'].tolist()
        vals = []
        for x in labels:
            if type(x) == float:
                vals.append(x)
            else:
                vals.append(max(set(x),key=x.count))

        vals = self.labelenc.transform(vals)

        return vals

    def run_model(self, x, y, train=True):
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        if train:
            print("train start")
            self.clf.fit(x_train,y_train)
            print("training finished")
        y_pred = self.clf.predict(x_test)
        print(accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))



    @staticmethod
    def clean_str(txt):
        text = ''.join([w for w in txt if w not in string.punctuation])
        text = text.lower()
        return text

class SVM_supconf():
    def __init__(self):
        pass

def run():
    saveTloadF = True
    # if input("save or load or arg").lower() == "save":
    #     saveTloadF = True
    path = getcwd()
    path += "/ArgumentAnnotatedEssays-2.0/"
    data = Reader(path).load_from_directory()

    # vec = None
    # clf = None
    # if not saveTloadF:
    #     name = input("Enter Model name with file extension")
    #     with open(name, 'rb') as r:
    #         vec, clf = joblib.load(r)
    #
    # ml_model = SVM_argClassification(path, datareader=Reader(path), vec=vec, clf=clf, build=False)
    #
    # print("Training model")
    # if saveTloadF:
    #     ml_model.train_arg()
    #
    # print("Predicting")
    # y_pred = ml_model.test_arg()
    #
    # print("Scoring")
    # ml_model.score(y_pred)
    # # if saveTloadF:
    # #     name = input("Enter model save_name:")
    # #     ml_model.vectoriser_mod.stop_words_ = None
    # #     with open(f'{name}.arg', 'wb') as r:
    # #         joblib.dump((ml_model.vectoriser_mod, ml_model.clf), r)
    #
    # x_train = ml_model.x_train
    # y_train = ml_model.y_train
    # x_test = ml_model.x_test
    # y_test = ml_model.y_test
    # # overall_data = ml_model.overall_data  # formatted data
    # all_sentences = ml_model.all_sents
    # flatten = iter(chain(*all_sentences))
    # for i, x in enumerate(y_pred):


    # relational classifier
    svm_relation = SVM_Relationship(data, build=True)

    svm_relation.build_relations()

    x = svm_relation.get_features()
    y = svm_relation.get_labels()

    svm_relation.run_model(x=x, y=y, train=True)

    # return x_train, y_train, x_test, y_test, y_pred

    return 1,1,1,1,1