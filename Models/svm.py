import string

import nltk

from dataReader.dataReader import Reader, get_train_test_split
from nltk.tokenize import RegexpTokenizer, sent_tokenize
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

        x_train = self.__prep_data(self.train_dataset,"train")
        y_train = train_list
        x_test = self.__prep_data(self.test_dataset, "test")
        y_test = test_list
        # print(x_train)
        raise ValueError
        return x_train, y_train, x_test, y_test

    @staticmethod
    def __prep_data(dataset, typ= "train") -> pd.DataFrame:
        #
        print(f"Prepping {typ} Dataset")
        # print(dataset.columns.values)
        # print(dataset.iloc[0])
        # dataset.drop(['fileid','label'], axis=1, inplace=True)
        # print(dataset.columns.values)
        punct_rem_tokeniser = RegexpTokenizer(r'\w+')
        # texts = dataset.loc[:, ["fullText1","fileid"]]
        texts = dataset.fullText1.unique()
        all_texts = []

        labels = []

        print("Prepping features")
        for text in texts:
            manip_text = text
            manip_text = manip_text.lower()

            # remove full stops
            sentences_split = sent_tokenize(manip_text)
            sentences_punt = [x[:-1] for x in sentences_split]

            # get number of punctuations marks in each sentence
            num_puncts_lst = []

            for punct in sentences_punt:
                num_punct = sum([ 1 for char in punct if char in string.punctuation])
                num_puncts_lst.append(num_punct)

            # retrieve sentence length in term of only alphabet characters
            sent_len_lst = []
            sent_unpunct = []
            for punct in sentences_punt:
                non_punct = punct_rem_tokeniser.tokenize(punct)
                sent_unpunct.append(non_punct)
                sent_len = len(non_punct)
                sent_len_lst.append(sent_len)

            # 3 successive words either side
            prev_words = []
            after_words = []
            for j in range(len(sent_unpunct)-1):
                before = "."
                after = "."
                if j != 0:
                    before = sent_unpunct[j-1][-3:]
                if j < len(texts) - 1:
                    after = sent_unpunct[j+1][:3]
                prev_words.append(before)
                after_words.append(after)

            # raise ValueError
            pos_tagged = nltk.pos_tag_sents(sent_unpunct)

            # nltk pos tag_word?
            overall_sentences = list(map(list,zip(sentences_punt,num_puncts_lst,sent_len_lst, prev_words, after_words, pos_tagged)))
            [all_texts.append([x,y,z,a,b,c]) for x,y,z,a,b,c in overall_sentences]

        print(len(all_texts))
        print(all_texts[0])
        columns = ["punct_sentences","number_punct","sentence_length", "prev", "after", "pos_tag"]
        dataframe = pd.DataFrame(all_texts)
        dataframe.columns = columns
        print(dataframe.shape)
        print(dataframe.head())
        return dataframe

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
