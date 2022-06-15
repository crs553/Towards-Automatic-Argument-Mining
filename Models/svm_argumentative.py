import string

import nltk

from dataReader.dataReader import Reader, get_train_test_split
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from sklearn import svm
import pandas as pd
import string
from os import getcwd


class SVM_Argumentative():
    def __init__(self, path):
        """
        Support Vector Machine initialisation
        :param path:
        """
        self.datareader = Reader(path)

        self.path = path

        self.x_train, self.y_train, self.x_test, self.y_test = self.prep_data()
        # self.target = (0, 1, 2)  # 0 is not a major claim, 1 is a major claim
        self.is_trained = False

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

        #prep test and train splits
        x_train, y_train = self.__prep_data(self.train_dataset, "train")
        x_test, y_test = self.__prep_data(self.test_dataset, "test")
        print("Features added")
        return x_train, y_train, x_test, y_test

    @staticmethod
    def __prep_data(dataset, typ="train") -> pd.DataFrame:
        #
        print(f"Prepping {typ} Dataset")
        punct_rem_tokeniser = RegexpTokenizer(r'\w+')
        texts = dataset.fullText1.unique()
        all_texts = []
        for text in texts:
            manip_text = text
            manip_text = manip_text.lower()

            # remove full stops
            sentences_split = sent_tokenize(manip_text)
            sentences_punt = [x[:-1] for x in sentences_split]

            # get number of punctuations marks in each sentence
            num_puncts_lst = []

            for punct in sentences_punt:
                num_punct = sum([1 for char in punct if char in string.punctuation])
                num_puncts_lst.append(num_punct)

            # retrieve sentence length in term of only alphabet characters
            sent_unpunct = [punct_rem_tokeniser.tokenize(x) for x in sentences_punt]
            sent_len_lst = [len(x) for x in sent_unpunct]

            # 3 successive words either side
            prev_words = [x[-3:] for x in sent_unpunct]
            prev_words.insert(0, ["."])
            prev_words = prev_words[:-1]
            after_words = [x[:3] for x in sent_unpunct]
            after_words.insert(-1, ["."])
            after_words = after_words[1:]

            # for

            # raise ValueError
            pos_tagged = nltk.pos_tag_sents(sent_unpunct)

            # nltk pos tag_word?
            overall_sentences = list(map(list,
                                         zip(sentences_split, sentences_punt, num_puncts_lst, sent_len_lst, prev_words,
                                             after_words, pos_tagged)))
            [all_texts.append(list(x)) for x in overall_sentences]

        # print(len(all_texts))
        # print(all_texts[0])
        columns = ["unformat_sentence", "punct_sentences", "number_punct", "sentence_length", "prev", "after",
                   "pos_tag"]
        dataframe = pd.DataFrame(all_texts)
        dataframe.columns = columns
        print(f"{typ} dataset shape: {dataframe.shape}")

        # label creation
        arg_sentences = set()
        [arg_sentences.add(sent.lower()) for sent in dataset.originalArg1.unique()]
        [arg_sentences.add(sent.lower()) for sent in dataset.originalArg2.unique()]

        all_sent = dataframe['unformat_sentence'].to_list()
        labels = []
        for sent in all_sent:
            if sent.lower() in arg_sentences:
                labels.append(1)
            else:
                labels.append(0)
        # labels = [y for x in dataset['unformat_sentence'].to_list() if x in arg_sentences]
        # print(len(labels))

        print(f"argumentative sentence amount: {len(arg_sentences)}")
        print(f"label total: {len(labels)}")
        # print(dataset.columns.values)
        return dataframe, labels

    def train(self):
        # prepped_trainset =
        print("t")
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

    # ml_model.train()


# print(arg1 + "\n" + arg2)

if __name__ == '__main__':
    pass
