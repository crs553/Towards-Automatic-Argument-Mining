import string

import nltk

from dataReader.dataReader import Reader, get_train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn import svm
import pandas as pd
import string
from os import getcwd


class SVM():
    def __init__(self, path):
        """
        Support Vector Machine initialisation
        :param path:
        """
        self.datareader = Reader(path)

        self.path = path

        self.train_dataset = pd.DataFrame()
        self.test_dataset = pd.DataFrame()
        self.prep_data()
        # self.target = (0, 1, 2)  # 0 is not a major claim, 1 is a major claim
        self.is_trained = False

    def prep_data(self):
        """PReps the training a test data"""
        train_test_split = get_train_test_split(self.path)
        data = self.datareader.load_from_directory()
        print("Directory loaded")
        ### NOTE: difference in number of unique texts vs number of fileid -- dataset error?
        # print(data['fullText1'].nunique())
        # print(data['fileid'].nunique())

        train_list = [i for i, (_, y) in enumerate(train_test_split) if y == 0]
        test_list = [i for i, (_, y) in enumerate(train_test_split) if y == 1]
        # test_list = [x-1 for x in test_list]
        # train_list = [x-1 for x in train_list]

        pd.set_option("display.max_columns", None)

        self.train_dataset = data[data['fileid'].isin(train_list)].copy()
        self.test_dataset = data[data['fileid'].isin(test_list)].copy()
        if len(self.test_dataset) + len(self.train_dataset) != len(data):
            raise ValueError("test and training csv does not contain all files")

        x_train = self.__prep_data(self.train_dataset)
        y_train = train_list
        x_test = self.__prep_data(self.test_dataset)
        y_test = test_list
        # print(x_train)
        raise ValueError
        return x_train, y_train, x_test, y_test

    @staticmethod
    def __prep_data(dataset):
        #
        # print("Prepping Train Dataset")
        # print(dataset.columns.values)
        # print(dataset.iloc[0])
        # dataset.drop(['fileid','label'], axis=1, inplace=True)
        # print(dataset.columns.values)
        tokeniser = RegexpTokenizer(r'\w+')
        # texts = dataset.loc[:, ["fullText1","fileid"]]
        print(dataset.shape)
        texts = dataset.fullText1.unique()
        all_texts = []

        for i, text in enumerate(texts):
            manip_text = text
            manip_text = manip_text.lower()

            # remove full stops
            sentences_punt = manip_text.split(". ")
            sentences_punt[-1] = sentences_punt[-1][:-1]

            # get number of punctuations marks in each sentence
            num_of_puncts = []

            for punct in sentences_punt:
                num_punct = sum([ 1 for char in punct if char in string.punctuation])
                num_of_puncts.append(num_punct)


            # retrieve sentence length in term of only alphabet characters
            sentences_non_punct = []
            for punct in sentences_punt:
                non_punct =
                sentences_non_punct.append()
                break


            # 3 successive words either side
            before = None
            after = None

            #pos tag
            # nltk.pos_tag_sents()
            # nltk pos tag_word?
            overall_sentences = list(zip(sentences_punt,num_of_puncts))
            all_texts.append((i,overall_sentences))

        for _, sent in all_texts:
            print(sent[-1])
            break
        raise ValueError("Hello")

        # # # obtained flattened list from panda columns
        # # full_texts = texts.values.tolist()
        # # full_texts = [x for xs in full_texts for x in xs]
        # print(len(full_texts))
        # no_dup_texts = list(dict.fromkeys(full_texts))
        # print(len(no_dup_texts))
        # # raise ValueError()
        # new_dataset = pd.unique(dataset[dataset["fileText1"]])
        # print(len(new_dataset))
        # raise FileNotFoundError("This is a new test")
        return dataset

    def __prep_test_data(self, t_list):
        data = "1"
        return data

    def train(self):
        prepped_trainset = self.change_labels(dataset=self.train_dataset)
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

    ml_model = SVM(path)

    # ml_model.train()


# print(arg1 + "\n" + arg2)

if __name__ == '__main__':
    pass
