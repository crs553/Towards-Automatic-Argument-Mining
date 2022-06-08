from dataReader.dataReader import Reader
from os import getcwd
import re
from nltk import sent_tokenize
import string
import json


class disInd():
    def __init__(self, texts=None):
        self.__indicators = self.open_indicators()
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
    def open_indicators() -> list:
        """
        Opens the discourse indicators folder and returns those indicators
        :return file: list organised into support and conflict indicators they are indicated by 0 and 1 respectively
        """
        cwd = getcwd()
        f = open(cwd + "/Models/combined_indicators.txt", "r")
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
        # print(arg1 + "\n" + arg2)
        for indicator in self.__indicators:
            len_ind = len(indicator)
            # print("indicator: "+indicator)
            # print(arg1[-len_ind:])
            # print(arg2[:len_ind])
            if arg2[:len_ind] == indicator or arg1[-len_ind:] == indicator:
                return True
        return False
        # words += str(i) for (i,_) in self.__indicators
        # for (word, ind_type) in self.__indicators:
        #     print(word)
        #     text = text.split(word)

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


def main():
    path = "/home/charlie/Documents/Project/ArgumentAnnotatedEssays-2.0/"
    reader = Reader(path)

    # t = reader.load_from_directory()
    total = 0
    v = disInd()

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

    print(f"Precision: {precision_numerator/total_identified}")
    print(f"Recall over full text {total_identified / total}")


if __name__ == "__main__":
    main()
