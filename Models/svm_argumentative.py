import string

import nltk
from dataReader.dataReader import Reader, get_train_test_split
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm
import pandas as pd
import string
from os import getcwd
import numpy as np

from sklearn.preprocessing import LabelEncoder
import gensim
from gensim.models import Word2Vec
from nltk.corpus import wordnet


class SVM_Argumentative():
    def __init__(self, path):
        """
        Support Vector Machine initialisation
        :param path:
        """
        self.datareader = Reader(path)

        self.path = path
        self.embed_model = None

        self.x_train, self.y_train, self.x_test, self.y_test = self.prep_data()
        # self.target = (0, 1, 2)  # 0 is not a major claim, 1 is a major claim
        self.is_trained = False
        self.vectoriser = None

    def prep_data(self):
        """PReps the training a test data"""
        train_test_split = get_train_test_split(self.path)
        data = self.datareader.load_from_directory()
        print("Directory loaded")

        train_list = [i for i, (_, y) in enumerate(train_test_split) if y == 0]
        test_list = [i for i, (_, y) in enumerate(train_test_split) if y == 1]

        pd.set_option("display.max_columns", None)
        self.train_dataset = data[data['fileid'].isin(train_list)].copy()
        self.test_dataset = data[data['fileid'].isin(test_list)].copy()
        if len(self.test_dataset) + len(self.train_dataset) != len(data):
            raise ValueError("test and training csv does not contain all files")

        # prep test and train splits
        text_data = data.fullText1.unique()
        # print(text_data[0])
        # text_data = [sent_tokenize(x.lower()) for x in text_data]

        self.vectoriser = TfidfVectorizer()
        self.vectoriser.fit(text_data)

        # lem_texts = []
        # punct_rem_tokeniser = RegexpTokenizer(r'\w+')
        # for text in text_data:
        #     manip_text = text.lower()
        #     sentences_split = sent_tokenize(manip_text)
        #     sentences_unpunct = []
        #     [sentences_unpunct.append(punct_rem_tokeniser.tokenize(x)) for x in sentences_split]
        #     lem_texts.append([self.lem_sentence(sent) for sent in sentences_unpunct])

        df = pd.DataFrame(text_data, columns=["texts"])
        # df.transpose()
        # df.columns = ["lem_texts"]
        # print(df.head())


        #
        # lems = [self.vectoriser.transform(x).to for x in lem_texts]
        # # vector = self.vectoriser.transform([lem_texts[0]])
        # print(vector.shape)
        # print(list(vector))
        # df = pd.DataFrame(data = lems, columns=["LemmatisedSentences"])
        # print(df.tail)

        # raise ValueError

        # print(text_data)
        w2v = Word2Vec(min_count=1, window = 2, alpha=0.03)
        w2v.build_vocab(sentences = text_data)
        w2v.train(sentence=corpus, total_examples = w2v.corpus_count, epochs=10, workers = 5)


        # self.embed_model = gensim.models.KeyedVectors
        # self.embed_model.load_word2vec_format("/home/charlie/Documents/Project/word2vec-google-news-300",
        #                                       binary=True)
        # self.embed_model = Word2Vec(text_data, min_count=1, vector_size=len(text_data), workers=3, window=3,sg=1)
        # self.embed_model.train(text_data, total_examples=len(text_data), epochs=5)
        # dis = self.cosine_distance(self.embed_model, text_data[0], text_data,5)
        # print(dis)

        x_train, y_train = self.__prep_data(self.train_dataset, "train")
        x_test, y_test = self.__prep_data(self.test_dataset, "test")
        print("Features added")
        return x_train, y_train, x_test, y_test

    # def __encode_one_hot(self):
    #     encoding = OneHotEncoder()

    def __prep_data(self, dataset, typ="train") -> pd.DataFrame:
        #
        print(f"Prepping {typ} Dataset")
        punct_rem_tokeniser = RegexpTokenizer(r'\w+')
        texts = dataset.fullText1.unique()

        # def word

        sentences_split = []


        lem_texts = []
        for text in texts:
            manip_text = text.lower()
            sentences_split = sent_tokenize(manip_text)
            sentences_unpunct = []
            [sentences_unpunct.append(punct_rem_tokeniser.tokenize(x)) for x in sentences_split]
            lem_texts.append([self.lem_sentence(sent) for sent in sentences_unpunct])

        print(lem_texts)
        raise ValueError


        #     manip_text = text.lower()
        #     print(manip_text)
        #     raise ValueError()
        #
        #     # remove full stops
        #     sentences_split_temp = sent_tokenize(manip_text)
        #     sentences_split.append(sentences_split_temp)
        #     sentences_punt = [x[:-1] for x in sentences_split_temp]
        #
        #     # get number of punctuations marks in each sentence
        #     num_puncts_lst = []
        #     for punct in sentences_punt:
        #         num_punct = sum([1 for char in punct if char in string.punctuation])
        #         num_puncts_lst.append(num_punct)
        #
        #     # retrieve sentence length in terms of only alphabet characters
        #     sent_unpunct = [punct_rem_tokeniser.tokenize(x) for x in sentences_punt]
        #     sent_len_lst = [len(x) for x in sent_unpunct]
        #
        #     # 3 successive words either side
        #     prev_words = [x[-3:] for x in sent_unpunct]
        #     prev_words.insert(0, ["."])
        #     prev_words = prev_words[:-1]
        #     after_words = [x[:3] for x in sent_unpunct]
        #     after_words.insert(-1, ["."])
        #     after_words = after_words[1:]
        #
        #     # raise ValueError
        #     pos_tagged = nltk.pos_tag_sents(sent_unpunct)
        #     pos_tagged = [x for j in pos_tagged for (_, x) in j]
        #     # new_pos = []
        #
        #     # for j in pos_tagged:
        #     #     lst = list(map(ord, j))
        #     # new_pos.append(lst)
        #     # print(new_pos)
        #     # raise FileNotFoundError
        #
        #     # nltk pos tag_word?
        #     # print(sentences_split)
        #     # raise FileNotFoundError
        #
        #     # encode data using ascii
        #
        #     overall_sentences = list(map(list,
        #                                  zip(sentences_punt, num_puncts_lst, sent_len_lst, prev_words,
        #                                      after_words, pos_tagged)))
        #     [all_texts.append(list(x)) for x in overall_sentences]
        #
        # # print(len(all_texts))
        # # print(all_texts[0])
        # columns = ["punct_sentences", "number_punct", "sentence_length", "prev", "after",
        #            "pos_tag"]
        # dataframe = pd.DataFrame(all_texts)
        # dataframe.columns = columns
        # dataframe.unformat_sentence.apply()
        # print(f"{typ} dataset shape: {dataframe.shape}")
        #
        # # label creation
        # arg_sentences = set()
        # [arg_sentences.add(sent.lower()) for sent in dataset.originalArg1.unique()]
        # [arg_sentences.add(sent.lower()) for sent in dataset.originalArg2.unique()]
        #
        # all_sent = dataframe['unformat_sentence'].to_list()
        # labels_lst = []
        # for sent in all_sent:
        #     if sent.lower() in arg_sentences:
        #         labels_lst.append(1)
        #     else:
        #         labels_lst.append(0)
        # labels = pd.DataFrame(labels_lst)
        # labels.columns = ["label"]
        #
        # # labels = [y for x in dataset['unformat_sentence'].to_list() if x in arg_sentences]
        # # print(len(labels))
        #
        # print(f"argumentative sentence amount: {len(arg_sentences)}")
        # print(f"label total: {labels.shape[0]}")
        # print(f"number of arg sentences {len([x for x in labels_lst if x == 1])}")
        # print(f"number of non-arg sentences {len([x for x in labels_lst if x == 0])}")
        # # print(dataset.columns.values)
        dataframe = None
        labels = None
        return dataframe, labels


    @staticmethod
    def lem_sentence(sent):
        """Adapted from https://gaurav5430.medium.com/using-nltk-for-lemmatizing-sentences-c1bfff963258"""
        lemmatizer = WordNetLemmatizer()

        tagged = nltk.pos_tag(sent)
        wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), tagged)

        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                # if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:
                # else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        return " ".join(lemmatized_sentence)

    def train(self):
        # prepped_trainset =
        x_train = self.x_train.apply(LabelEncoder.fit_transform)
        svm_arg = svm.SVC(kernel='linear')
        svm_arg.fit(self.x_train, self.y_train)
        return None
        # for x in prepped_trainset:
        #     print(x['label'], end= " ")
        #     # break
        # self.is_trained = True
        # return None

    @staticmethod
    def change_labels(bidirect=False, dataset=None):
        if dataset is None:
            raise ValueError()

        if bidirect:
            raise ValueError("Bidirection is True is not implemented")

        dataset[dataset['label'] == -1.0] = 1.0
        dataset[dataset['label'] == -2.0] = 2.0


def run():
    path = getcwd()
    path += "/ArgumentAnnotatedEssays-2.0/"

    ml_model = SVM_Argumentative(path)

    print("Training model")
    ml_model.train()


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# print(arg1 + "\n" + arg2)

if __name__ == '__main__':
    pass
