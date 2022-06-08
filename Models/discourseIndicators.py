from dataReader.dataReader import Reader
from os import getcwd
import re
from nltk import sent_tokenize
import string
import json


class disInd():
    def __init__(self, texts=None, indicators="premise_indicators"):
        self.__indicators = self.open_indicators(indicators)
        self.__texts = texts

    def set_text(self, text):
        self.__texts = text

    @property
    def indicators(self) -> list:
        """Returns the indicators obtained from the json file"""
        return self.__indicators

    @property
    def texts(self) -> list:
        """ returns list of texts """
        return self.__texts

    @staticmethod
    def open_indicators(text_file) -> list:
        """
        Opens the discourse indicators folder and returns those indicators
        :param self:
        :return file: list organised into support and conflict indicators they are indicated by 0 and 1 respectively
        """
        cwd = getcwd()
        print(f"Loading indicators file: {text_file}.txt")
        f = open(cwd + f"/Models/{text_file}.txt", "r")

        file = f.read()
        f.close()
        file = file.split("\n")
        # print(file)
        return file

    def run(self) -> (int, [str, str]):
        correct = 0
        text = self.__texts[0]['fullText1']
        sent_text = sent_tokenize(text)
        identified_tuples = []
        for i in range(len(sent_text) - 1):
            arg1 = sent_text[i]
            arg2 = sent_text[i + 1]
            if self.compare_args(arg1, arg2):
                correct += 1
                # appends arg1, arg2 and the location of the first argument
                # in terms of the number of sentences
                identified_tuples.append((arg1, arg2))
        return correct, identified_tuples

    def compare_args(self, arg1, arg2):
        translator = str.maketrans('', '', string.punctuation)
        arg1 = arg1.lower()
        arg2 = arg2.lower()
        arg1 = arg1.translate(translator)
        arg2 = arg2.translate(translator)
        for indicator in self.__indicators:
            len_ind = len(indicator)
            if arg2[:len_ind] == indicator or arg1[-len_ind:] == indicator:
                return True
        return False

    def compare_identifications(self, identifications):
        correct = 0
        for id1, id2 in identifications:
            for x in self.__texts:
                arg1 = x['originalArg1']
                arg2 = x['originalArg2']
                if arg1 == id1 and arg2 == id2:
                    correct += 1
                    break
        return correct


def run():
    path = "/home/charlie/Documents/Project/ArgumentAnnotatedEssays-2.0/"
    reader = Reader(path)

    # t = reader.load_from_directory()
    total = 0
    indicators = "combined_indicators" # change this to change the text file used
    v = disInd(None, indicators)

    # case ran over full text
    total_identified = 0
    identifications = []

    precision_numerator = 0

    # For case argument it already split
    for i in range(402):
        t = reader.load_single_file(i)
        # print(t[0])
        v.set_text(t)
        # print(t[0]['fullText1'])

        total += len(t)
        currc, currid = v.run()
        total_identified += currc

        precision_numerator += v.compare_identifications(currid)

    precision = precision_numerator/total_identified
    recall = total_identified/total
    f1 = (precision*recall)/(precision+recall)
    print(f"Values over while dataset")
    print(f"Precision\t{precision}")
    print(f"Recall\t{recall}")
    print(f"f1\t{f1}")


if __name__ == "__main__":
    run()
