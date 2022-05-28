from glob import glob
from os import getcwd

import numpy as np
import pandas as pd
import xmltodict
from nltk.tokenize import  sent_tokenize,word_tokenize


class Reader():
    def __init__(self, path: str) -> None:
        """
        Initialisation of the reader
        :param path: the path to the folder containing the unzipped files
        """

        self.__path = path
        self.__test_train_split = self.__setup_train_test()
        self.__texts = self.__open_text()
        self.__anns = self.__load_annotations()

    @property
    def path(self) -> str:
        return self.__path

    @property
    def test_train_list(self) -> list:
        return self.__test_train_split

    @property
    def annotations(self) -> list:
        return self.__anns

    @property
    def texts(self) -> list:
        return self.__texts

    def __setup_train_test(self) -> list:
        """
        Finds train and test type for each file splits and returns them as list
        :return:
        """
        f = open(self.__path + "/train-test-split.csv", "r")
        file = f.read()
        f.close()
        clean_file = file.split("\n")
        clean_file = clean_file[1:-1]
        splits = []
        for f_and_t in clean_file:
            a = f_and_t[1:-1].split(";")
            splits.append((a[0][:-1], a[1][1:]))

        return splits

    def __open_text(self) -> list:
        """Open all text files within brat-project-final
        :return: list of found text files"""
        local_texts = []
        filenames = [i for i in glob(self.__path + "/brat-project-final/*.txt")]
        filenames.sort()

        for fname in filenames:
            f = open(fname, "r")
            file = f.read()
            f.close()
            local_texts.append((fname[-12:], file))

        return local_texts

    def __load_annotations(self) -> list:
        """
        Opens all annotation files within brat-project-final and creates a relational matrix
        :return:  list of found annotation xml type files
        """
        print("Directory annotation loading...")
        print("Detected files: ")

        anns = []

        #change for xml needed
        filenames = [i for i in glob(self.__path + "/brat-project-final/*ann.xml")]
        filenames.sort()

        for i, fname in enumerate(filenames):
            print(fname[-16:])
            f = open(fname, "r")
            file = f.read()
            f.close()
            anns.append((fname[-16:], xmltodict.parse(file)))

        return anns

    @staticmethod
    def load_adu(file, fileid):
        """

        adapted from: https://github.com/negedng/argument_BERT/blob/master/preprocessing/data_loader.py
        :param file:
        :param fileid:
        :return:
        """
        # relations in the form of a matrix
        relations = {}

        # argumentation id
        argId = fileid

        # parse the xml data to dictionary
        xmlData = xmltodict.parse(file)

        # get length of one matrix
        matrixLen = len(xmlData['Annotation']['Proposition'])
        relationCount = 0

        # total number of possibe relations
        totalRelation = matrixLen * matrixLen

        # define the matrix
        relationMatrix = np.zeros((matrixLen,matrixLen))

        # original text for file after \n removed and token
        original_txt = ""
        sent_tokenize_list = None
        len_sent_token_list = None

        if 'OriginalText' in xmlData['Annotation']:
            original_text = xmlData['Annotation']['OriginalText']
            original_txt = original_text.replace('\n', ' ')
            sent_tokenize_list = sent_tokenize(original_text)



if __name__ == "__main__":
    path = getcwd()[:-11] + "/ArgumentAnnotatedEssays-2.0"
    reader = Reader(path)
    splits = reader.annotations

    print(splits[1])

    # for a,b in splits:
    #     print(f"{a},{b}")
    # print(splits[401])
    # print(len(splits))
