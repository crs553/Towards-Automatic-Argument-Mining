from dataReader.dataReader import Reader
from os import getcwd
import re
import json


class disInd():
    def __init__(self, texts: list):
        self.__indicators = self.open_indicators()
        self.__texts = texts

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
        file = json.load(open(cwd + "/discourse_indicators.json", "r"))
        vals = list(zip(file['support'], [0])) + list(zip(file['conflict'], [1]))
        len_support = len(file['support'])
        len_conflict = len(file['conflict'])
        support = list(zip(file['support'], [0] * len_support))
        conflict = list(zip(file['conflict'], [1] * len_conflict))
        vals = [*support, *conflict]
        print(vals)
        return vals

    def run_single_file(self, id):
        text = self.__texts[id]
        text = text.lower()
        for (word, ind) in self.indicators:
            character = "¬" if ind == 1 else "~"
            text = text.replace(word + " ",f'{character}')
        print(text)
        text = re.split(' ~| ¬|¬|~', text)
        text = [t for t in text if t != '']
        print(text)

        # words += str(i) for (i,_) in self.__indicators
        # for (word, ind_type) in self.__indicators:
        #     print(word)
        #     text = text.split(word)



if __name__ == "__main__":
    v = disInd(["This is a test since I unsure because texts except I want to see if this works no it does not nonetheless it might"])
    v.run_single_file(0)
