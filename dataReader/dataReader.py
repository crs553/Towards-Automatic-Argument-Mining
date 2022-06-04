from glob import glob
from os import getcwd

import numpy as np
import pandas as pd
import xmltodict
from nltk.tokenize import sent_tokenize, word_tokenize


class Reader():
    def __init__(self, path: str) -> None:
        """
        Initialisation of the reader
        :param path: the path to the folder containing the unzipped files
        """

        self.__path = path
        print("Loading files")
        self.filenames = [i for i in glob(self.__path + "/brat-project-final/*.txt")]
        print(f"Detected {len(self.filenames)} files:")
        self.filenames.sort()
        for num, i in enumerate(self.filenames):
            t = "\n" if num % 10 == 0 else ""
            print(f"{i[-12:]}\t", end= t)
        print()
        self.__test_train_split = self.__setup_train_test()
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

    def open_text(self, file_id) -> (str,str):
        """Open all text files within brat-project-final
        :param file_id: index of the file to load
        :return: tuple of filename and the file contents"""
        local_text = None

        fname = self.filenames[file_id]
        f = open(fname, "r")
        file = f.read()
        f.close()
        local_text = (fname[-12:], file)

        if not local_text:
            raise ValueError("Could not load file")

        return local_text

    def __load_annotations(self) -> list:
        """
        Opens all annotation files within brat-project-final and creates a relational matrix
        :return:  list of found annotation xml type files
        """
        print("Directory annotation loading...")


        anns = []

        for i, fname in enumerate(self.filenames):
            print(fname[-16:])
            f = open(fname, "r")
            file = f.read()
            f.close()
            print(fname[-16:])
            print("\n\n"+file)
            anns.append((fname[-16:], xmltodict.parse(file)))

        return anns

    def load_adu(self, fileid):
        """

        adapted from: https://github.com/negedng/argument_BERT/blob/master/preprocessing/data_loader.py
        :param fileid:
        :return:
        """
        # relations in the form of a matrix
        relations = {}
        file_data = list()

        with open(self.filenames[fileid], 'r') as file_dat:
            file = file_dat.read()

        # argumentation id
        arg_id = fileid

        # parse the xml data to dictionary
        xml_data = xmltodict.parse(file)

        # get length of one matrix
        matrix_len = len(xml_data['Annotation']['Proposition'])
        relation_count = 0

        # total number of possibe relations
        total_relation = matrix_len * matrix_len

        # define the matrix
        relation_matrix = np.zeros((matrix_len, matrix_len))

        # original text for file after \n removed and token
        original_txt = ""
        sent_tokenize_list = None
        len_sent_token_list = None
        sens = 0

        # get propositions
        props = xml_data['Annotation']['Proposition']

        # if the original text is in the data, get it and tokenize it
        if 'OriginalText' in xml_data['Annotation']:
            original_text = xml_data['Annotation']['OriginalText']
            original_txt = original_text.replace('\n', ' ')
            sent_tokenize_list = sent_tokenize(original_text)
            sens = len(sent_tokenize_list)

        for propId, currProp in enumerate(props):
            if currProp['ADU']['@type'] == 'conclusion':
                adu_type = 2
            elif currProp['ADU']['@type'] == 'claim':
                adu_type = 1
            elif currProp['ADU']['@type'] == 'premise':
                adu_type = 0
            else:
                raise ValueError("ADU type unrecognised",
                                 f"{currProp['ADU']['@type']} is the unrecognised type")

            arg1 = currProp['text']
            originalSentenceArg1 = arg1
            positArg1 = -1

            if currProp['TextPosition']['@start'] != '-1':
                for sentence in sent_tokenize_list:

                    if arg1 in sentence:
                        originalSentenceArg1 = sentence
                        sen1 = sent_tokenize_list.index(sentence)

                positArg1 = int(currProp['TextPosition']['@start'])
            line_data = {
                'argumentationID': arg_id,
                'arg1': arg1,
                'originalArg1': originalSentenceArg1,
                'label': adu_type,
                'fullText1': original_txt,
                'positArg1': positArg1 / len(original_txt),
            }
            file_data.append(line_data)
        return file_data


if __name__ == "__main__":
    path = getcwd()[:-11] + "/ArgumentAnnotatedEssays-2.0"
    # path = "/home/charlie/Documents/Project/ArgumentAnnotatedEssays-2.0/"
    reader = Reader(path)

    # for a,b in splits:
    #     print(f"{a},{b}")
    # print(splits[401])
    # print(len(splits))
