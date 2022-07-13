import string
from itertools import chain
from os import getcwd
import nltk
import numpy as np
import pandas as pd
import spacy
from imblearn.under_sampling import RandomUnderSampler
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, recall_score, \
    f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn_pandas import DataFrameMapper
from tqdm import tqdm
from dataReader.dataReader import Reader
import Models.discourseIndicators as discourseIndicators


class SVM_argClassification():
    def __init__(self, path, datareader=None, vec=None, clf=None, build=True, combined=False) -> None:
        """
        SVM for argumentative identification
        :param path: path to project directory
        :param datareader: datareader class for loading the intial argument maps for the class
        :param vec: cetoriser class
        :param clf: clf class for passing in pipeline
        :param build: option to load dataset prebuilt from class
        :param combined: turn on combined method
        """
        self.combined = combined
        self.features = None
        self.datareader = datareader
        self.build = build
        self.overall_data = {}
        self.all_sents = list()
        self.data = list()
        self.path = path
        self.embed_model = None
        self.vectoriser_mod = False if vec is None else vec
        self.mapper = None
        self.standardScaler = StandardScaler(with_mean=False)
        self.clf = clf if clf is not None else make_pipeline(self.standardScaler, svm.SVC(gamma='auto'))

        self.text_data = list()
        self.x_train_loc = list()
        self.x_test_loc = list()
        self.x_train, self.x_test, self.y_train, self.y_test = self.prep_data_arg()
        self.is_trained = False
        self.y_pred = None

    def prep_data_arg(self) -> tuple[list, list, list, list]:
        """
        Preps the test and training data for non-combined metho
        :return: tuple of list of the split up x y of the train and test set with their respective xs and ys
        """

        data = self.datareader.load_from_directory()

        self.data = data
        print("Directory loaded")

        # get all textual data
        text_data = data.fullText1.unique()
        pairs = []

        # split textual data into sentences
        all_sentences = [sent_tokenize(x.lower()) for x in text_data]
        pairs = []
        for i, unique in enumerate(text_data):
            pairs.append((self.data.loc[self.data['fullText1'] == unique, 'fileid'].iloc[0], all_sentences[i]))

        all_sent_fileid = [(i, clean_str(sent)) for i, sents in enumerate(all_sentences) for sent in sents]
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

            fileid = [x for x, _ in all_sent_fileid]

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
                  'before': all_before, 'after': all_after, 'y_vals': y_vals, 'fileid': fileid}

            df = pd.DataFrame(df)
            df['overall_index'] = df.index
            self.overall_data = df
            self.overall_data.to_pickle('arg_class_data.pkl')
            print("Model built")
        else:
            df = pd.read_pickle('arg_class_data.pkl')
            self.overall_data = df
            print("Model loaded")

        if self.vectoriser_mod is False:
            self.vectoriser_mod = TfidfVectorizer(stop_words="english", max_features=50, decode_error="ignore",
                                                  ngram_range=(1, 2))

        self.mapper = DataFrameMapper([('sent_vector', self.vectoriser_mod),
                                       ('before', self.vectoriser_mod),
                                       ('after', self.vectoriser_mod),
                                       ('pos_feature', None),
                                       ('puct_num', None),
                                       ('overall_index', None)
                                       ])

        y_vals = df['y_vals'].to_list()
        if self.combined:
            self.mapper.df_out = True
            self.features = self.mapper.fit_transform(df)
            x_train, x_test, y_train, y_test = train_test_split(self.features.tonumpy(), y_vals, shuffle=False,
                                                                test_size=0.3)
        else:
            self.features = self.mapper.fit_transform(df)
            x_train, x_test, y_train, y_test = train_test_split(self.features, y_vals)
        if self.combined:
            pass
        print("Features Created")
        return x_train, x_test, y_train, y_test

    def train_arg(self):
        """Training dta"""
        x_tr = self.x_train
        y_tr = self.y_train
        self.clf.fit(x_tr, y_tr)

        print("Training Complete")

    def test_arg(self):
        """Run testing on pipeline"""
        x_te = self.x_test
        self.y_pred = self.clf.predict(x_te)
        return self.y_pred

    def score(self, y_pred):
        """Score inputted prediction against the test set"""
        score(self.y_test, y_pred)

    def pred(self):
        """Return the predictions"""
        return self.y_pred


class SvmRelationship():
    def __init__(self, data, build=True, combined=False, x_train=None, x_test=None):
        """
        Initialisations of the relationships between sentence pairs
        :param data: input data from datareader
        :param build: if true build model else load from pkl file
        :param combined: create combined variables
        :param x_train: dataframe for train-set
        :param x_test: datagrame for test-set
        """

        #overall dataframe
        self.data = data

        #used in combined method
        self.combined = combined
        if combined:
            self.x_train = x_train.copy()
            self.x_test = x_test.copy()
            self.y_train = None
            self.y_test = None
            self.sentence_pairs = None

        #get unique labels
        labels = self.data.label.unique()
        labels = [abs(x) for x in labels.tolist()]
        # labels.extend([3.0])
        labels = list(set(labels))

        #fitting to encode labels
        self.labelenc = LabelEncoder()
        self.labelenc.fit([1.0, 0.0])

        #mapper/datamapper
        self.datamapper = None

        self.vectoriser_mod = TfidfVectorizer(stop_words="english", max_features=50, decode_error="ignore",ngram_range=(1, 2), min_df=1)
        self.mapper = DataFrameMapper([('sent1', self.vectoriser_mod),
                                       ('sent2', self.vectoriser_mod),
                                       ('three1', self.vectoriser_mod),
                                       ('three2', self.vectoriser_mod),
                                       ('senta_tag', self.vectoriser_mod),
                                       ('sentb_tag', self.vectoriser_mod),
                                       ('similarity', None),
                                       ('pos_a', None),
                                       ('pos_b', None),
                                       ('fileid', None)
                                       ])

        #build variable
        self.build = build

        #rlations
        self.relations = None

        #svm
        self.clf = make_pipeline(StandardScaler(with_mean=False), svm.SVC(gamma='auto'))

    def build_relations(self) -> None:
        """Build relations for non-combined dataset"""
        print("building relations database for all values")

        #get unique fullTexts
        fullTexts = self.data.fullText1.unique()

        #get all sentences as tokenised values
        all_sentences = [sent_tokenize(x) for x in fullTexts]

        #get all sentence pairs that are in the dataset
        pairs = []
        for i, unique in enumerate(fullTexts):
            pairs.append((self.data.loc[self.data['fullText1'] == unique, 'fileid'].iloc[0], all_sentences[i]))

        # for the positions of the pairs in the datsaet
        all_pos = [i for (i, _) in pairs]

        # get the file ids for all sentences
        all_sent_fileid = [(i, clean_str(sent)) for i, sents in enumerate(all_sentences) for sent in sents]

        #get the relation in the dataset
        relations = self.data.loc[:, ['originalArg1', 'originalArg2', 'label', 'fileid']]

        nlp = spacy.load("en_core_web_lg")
        if self.build:  # if dataset is not a pickle file create datsaet
            #create variables for each features in the dataset
            senta, sentb = [], []
            pos_tag_a = []
            pos_tag_b = []
            pos_a, pos_b = [], []
            sent_sim = []
            temp_file_pos = []
            labels = []
            three1, three2 = [], []

            # Progress bar
            pbar = tqdm(total=len(all_sentences))

            #for each document get the features of each sentence pair
            for i, doc in enumerate(all_sentences):
                len_doc = len(doc)
                pos = [(j / len_doc, x) for j, x in enumerate(doc)]
                prod = [(x, y) for x in doc for y in doc]
                for (sent1, sent2) in prod:

                    label_temp = self.data.loc[((self.data['originalArg1'] == sent1) &
                                                (self.data['originalArg2'] == sent2)), 'label'].tolist()
                    label_temp = [abs(x) for x in label_temp]
                    label_temp = list(set(label_temp))

                    # if the sentence pair is not contianed within the dataset
                    if len(label_temp) == 0:
                        label_temp = 3.0

                    #sentence similarity
                    nlps1 = nlp(sent1)
                    nlps2 = nlp(sent2)
                    sent_sim = nlps1.similarity(nlps2)

                    # pos tag of sentence
                    tag_a = [x.lower() for (_, x) in nltk.pos_tag(word_tokenize(sent1))]
                    tag_b = [x.lower() for (_, x) in nltk.pos_tag(word_tokenize(sent2))]
                    pos_tag_a.append(' '.join(tag_a))
                    pos_tag_b.append(' '.join(tag_b))

                    three1.append(sent1[-3:])
                    three2.append(sent2[:3])

                    labels.append(label_temp)
                    senta.append(sent1)
                    pos_a.append([a for (a, x) in pos if x == sent1][0])
                    sentb.append(sent2)
                    pos_b.append([a for (a, x) in pos if x == sent2][0])
                    temp_file_pos.append(all_pos[i])
                pbar.update(1)

            df = {"sent1": senta, "sent2": sentb, "fileid": temp_file_pos, "labels": labels,
                  'pos_a': pos_a, 'pos_b': pos_b, 'three1': three1, 'three2': three2, 'similarity': sent_sim,
                  'senta_tag': pos_tag_a, 'sentb_tag': pos_tag_b}
            self.relations = pd.DataFrame(df)
            if not self.combined:
                self.relations.to_pickle('relational_dataset.pkl')
        else:
            self.relations = pd.read_pickle('relational_dataset.pkl')
            print("Relational_dataset loaded")

    def build_relations_combined(self) -> None:
        """Combined relations datsaset builder"""
        # train_sents = self.x_train[self.x_train['sent_vectors']].to_list()
        # test_sents = self.x_test.copy()
        train_sents = self.x_train.copy()
        test_sents = self.x_test.copy()

        fullTexts = self.data.fullText1.unique()

        pairs = []
        all_sentences = [sent_tokenize(x) for x in fullTexts]

        for i, unique in enumerate(fullTexts):
            pairs.append((self.data.loc[self.data['fullText1'] == unique, 'fileid'].iloc[0], all_sentences[i]))

        all_pos = [i for (i, _) in pairs]

        all_sent_fileid = [(i, clean_str(sent)) for i, sents in enumerate(all_sentences) for sent in sents]
        all_fileid = [i for (i, _) in all_sent_fileid]

        relations = self.data.loc[:, ['originalArg1', 'originalArg2', 'label', 'fileid']]

        # sentences correctly classified as arguments are kept
        sent_keep_train = train_sents.loc[:, ['sent_vector']]
        sent_keep_test = test_sents.loc[:, ['sent_vector']]

        sent_train = []
        sent_test = []
        train_df = []
        fullTexts = self.data.fullText1.unique()
        all_sentences = [sent_tokenize(x) for x in fullTexts]

        df = pd.read_pickle("relational_dataset.pkl")

        labels_train = []
        labels_test = []
        index = df.index
        sent_lst_test = test_sents['sent_vector'].tolist()
        sent_lst_train = train_sents['sent_vector'].tolist()
        keep_pos = [-1] * len(index)

        sents_df = [df['sent1'][i].lower() for i in index]
        # keep_pos = [1 if s in sent_lst_test else 0 if s in sent_lst_train else -1 for s in sents_df]
        print(sent_lst_train)
        sents1 = df['sent1'].to_list()
        sents2 = df['sent2'].to_list()
        for i in index:

            s = sents1[i].lower()
            s2 = sents2[i].lower()
            if s in sent_lst_train or s2 in sent_lst_train:
                keep_pos[i] = 0

            elif s in sent_lst_test or s2 in sent_lst_test:
                keep_pos[i] = 1



        df['test_train_split'] = keep_pos

        data = df[df['test_train_split'] != -1].copy()

        self.relations = data

    def get_split_combined(self) -> tuple[list, list]:
        """split dataset into test and train set for combined dataset"""

        self.x_train = self.relations[self.relations['test_train_split'] == 0]
        self.x_test = self.relations[self.relations['test_train_split'] == 1]

        return self.x_train, self.x_test

    def get_features(self) -> object:
        """Get the features of a datset and return them"""
        if self.combined:
            self.mapper.fit_transform(self.relations)
            return self.mapper.transform(self.x_train), self.mapper.transform(self.x_test)
        features = self.mapper.fit_transform(self.relations)
        return features

    def get_labels(self) -> tuple[list, list]:
        """Get the labels within a dataset"""
        if self.combined:
            self.y_train = self.x_train['labels'].tolist()
            self.y_test = self.x_test['labels'].tolist()

            return self.__get_labels(self.y_train), self.__get_labels(self.y_test)

        labels = self.relations['labels'].tolist()
        return self.__get_labels(labels)

    def __get_labels(self, labels) -> list:
        """
        convert passed labels list into appropriately encoded list
        :param labels:
        :return:
        """
        vals = []
        for x in labels:
            if type(x) == float:
                vals.append(x)
            else:
                vals.append(float(max(set(x), key=x.count)))
        vals = [1 if x == any([0.0, 1.0, 2.0]) else 0 for x in vals]
        vals = self.labelenc.transform(vals)
        return vals

    def run_model(self, x=None, y=None, train=True):
        """
        Runs the model
        :param x: only required if training uncombined
        :param y: only required if traiing uncombined
        :param train: True or false value indicating want to train
        :return: predictions for labels
        """
        under_sampler = RandomUnderSampler(random_state=42)

        if self.combined:
            x_train = self.x_train
            x_test = self.x_test
            y_train = self.y_train
            y_test = self.y_test
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y)
        x_trnew, y_trnew = under_sampler.fit_resample(x_train, y_train)
        print(f"Train\nx: {len(x_trnew)}\ty: {len(y_trnew)}")
        if train:
            self.clf.fit(x_trnew, y_trnew)
        y_pred = self.clf.predict(x_test)
        return y_pred


def clean_str(txt:list) -> str:
    """
    Concatenates list to space separate lowercase string
    :param txt:list of strings
    :return: space separated lowercase str
    """
    text = ''.join([w for w in txt if w not in string.punctuation])
    text = text.lower()
    return text


def score(y_test, y_pred) -> None:
    """
    Prints score to terminal
    :param y_test: true values
    :param y_pred:  predicted values
    """
    print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 3)}\t"
          f"Recall: {round(recall_score(y_test, y_pred), 3)}\tF1 {round(f1_score(y_test, y_pred), 3)}")
    print(f"Micro report: {precision_recall_fscore_support(y_test, y_pred, average='micro')}")
    print("Report")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))


def run():
    """
    Runs the non-combined method
    :return:
    """

    #get current position of argument annotations dataset
    path = getcwd()
    path += "/ArgumentAnnotatedEssays-2.0/"
    reader = Reader(path)

    vec = None
    clf = None
    ml_model = None

    print("SVM argument Classification")
    ml_model = SVM_argClassification(path, datareader=Reader(path), vec=vec, clf=clf, build=True)
    ml_model.train_arg()
    y_pred = ml_model.test_arg()
    ml_model.score(y_pred)

    # SVM Relationship Classifier
    print("SVM Relationship Classifier")
    data = None
    if ml_model is None:
        data = reader.load_from_directory()
    else:
        data = ml_model.data

    svm_relation = SvmRelationship(data, build=False)
    svm_relation.build_relations()
    x = svm_relation.get_features()
    y = svm_relation.get_labels()
    svm_relation.run_model(x=x, y=y, train=True)


def run_combined():
    """
    Combined method
    :return:
    """
    path = getcwd()
    path += "/ArgumentAnnotatedEssays-2.0/"
    reader = Reader(path)
    # data = reader.load_from_directory()

    vec = None
    clf = None
    ml_model = None

    print("SVM argument Classification")
    ml_model = SVM_argClassification(path, datareader=Reader(path), vec=vec, clf=clf, build=True)

    x_test = ml_model.x_test
    x_train = ml_model.x_train
    y_train = ml_model.y_train

    ml_model.train_arg()
    y_pred = ml_model.test_arg()
    ml_model.score(y_pred)

    # SVM Relationship Classifier
    print("SVM Relationship Classifier")

    # Data Prep
    pred = ml_model.pred()
    data = ml_model.data
    dataframe = ml_model.overall_data
    features = ml_model.features
    test_data = prep_rel_dataset(pred, x_test, dataframe, test=True)

    train_data = prep_rel_dataset(y_train, x_train, dataframe, test=False)
    # raise ValueError
    svm_relation = SvmRelationship(data, combined=True, x_train=train_data, x_test=test_data)
    svm_relation.build_relations_combined()
    svm_relation.get_split_combined()

    x_test = svm_relation.x_test

    # Discourse Indicators
    y_pred_disc = discourseIndicators.run_combined(x_test)

    svm_relation.y_train, svm_relation.y_test = svm_relation.get_labels()

    # Score Discourse Indicators
    print("Scoring Discourse Indicators")
    score(svm_relation.y_test, y_pred_disc)
    svm_relation.x_train, svm_relation.x_test = svm_relation.get_features()
    y_pred = svm_relation.run_model()

    print("Link Classification Scoring")
    score(svm_relation.y_test, y_pred)

    print("Link and Discourse Combined Classification")
    overall_pred = []
    print()
    for i, x in enumerate(y_pred):
        if y_pred_disc[i] == 1 or y_pred[i] == 1:
            overall_pred.append(1)
        else:
            overall_pred.append(0)

    score(svm_relation.y_test, overall_pred)

    print("Finished")


def prep_rel_dataset(y, x, df, test=True):
    """
    prepares daaset for transition between argument classification and relation prediction
    :param y: y variables
    :param x: x variables
    :param df: dataframe of filtered data
    :param test:
    :return:
    """
    egs = len(x)
    if test:
        new_df = df[-egs:].copy()
    else:
        new_df = df[:egs].copy()

    # only keep values which are argumentative
    if test:
        new_df['y_pred'] = y
        new_df = new_df[new_df['y_pred'] == 1].copy()
        new_df.drop('y_pred', axis=1, inplace=True)
    else:
        new_df = new_df[new_df['y_vals'] == 1].copy()

    return new_df
