from dataReader.dataReader import Reader
from os import getcwd
import re
from nltk import sent_tokenize
import string
import json
from nltk.corpus import stopwords


class disInd():
    def __init__(self, texts=None, indicators="discourse_indicators", combined =False):
        """
        Intialises Discourse indicators
        :param texts: text to be ran on (can be set later using set_text method)
        :param indicators: name of .txt file to use for indicators
        """
        self.__indicators = self.open_indicators(indicators)
        if not combined:
            self.__texts = texts


    def set_text(self, text):
        """Sets texts for use in discoruse indicators"""
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
        :param text_file:
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
        """
        Runs the discourse indicator method at a sentence level
        :return: the number of correct instances followed by a list of tuples of the sentences with the correct identifications
        """
        correct = 0
        text = self.texts[0]['fullText1']
        sent_text = sent_tokenize(text)
        identified_tuples = []
        for i in range(len(sent_text)-1):
            arg1 = sent_text[i]
            arg2 = sent_text[i + 1]
            if self.compare_args(arg1, arg2):
                correct += 1
                # appends arg1, arg2 and the location of the first argument
                # in terms of the number of sentences
                identified_tuples.append((arg1, arg2))

        return correct, identified_tuples

    def run_combined(self, x_test):
        x_test.reset_index(drop=True, inplace=True)
        index = x_test.index
        pos = []
        sents1 = x_test['sent1'].tolist()
        sents2 = x_test['sent2'].tolist()

        for i in index:
            pos.append(1) if self.compare_args(sents1[i],sents2[i]) else pos.append(0)

        return pos



    def compare_args(self, arg1, arg2):
        """
        Compares arguments determining if the first argument ends with an indicator or the second starts with one
        :param arg1: The first sentence
        :param arg2: The second sentence
        :return bool: True if indicator is contained False if not
        """
        arg1 = self.format(arg1)
        arg2 = self.format(arg2)

        for indicator in self.__indicators:
            len_ind = len(indicator)
            if arg2[:len_ind] == indicator or arg1[-len_ind:] == indicator:
                return True
        return False

    @staticmethod
    def format(arg):
        """Removes punctuations and uppercase from passed string"""
        translator = str.maketrans('', '', string.punctuation)
        arg = arg.lower()
        arg = arg.translate(translator)
        return arg

    def compare_identifications(self, identifications):
        """Compares tuples of identified arguments with the arguments presented in text"""
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
    """Runs the discourse indicators model"""
    path = "/home/charlie/Documents/Project/ArgumentAnnotatedEssays-2.0/"
    reader = Reader(path)

    # t = reader.load_from_directory()
    total = 0
    indicators = "premise_indicators"  # change this to change the text file used
    v = disInd(None, indicators)

    # case ran over full text
    total_identified = 0

    precision_numerator = 0

    # For case argument it already split
    print("Running argument discourse")
    for i in range(402):
        t = reader.load_single_file(i)
        v.set_text(t)

        total += len(t)
        currc, currid = v.run()
        total_identified += currc

        precision_numerator += v.compare_identifications(currid)

    precision = precision_numerator / total_identified
    recall = total_identified / total
    f1 = (precision * recall) / (precision + recall)
    print(f"Values over while dataset")
    print(f"Precision\t{precision}")
    print(f"Recall\t{recall}")
    print(f"f1\t{f1}")

def run_combined(x_test):
    v = disInd(combined=True)
    return v.run_combined(x_test)



if __name__ == "__main__":
    run()
