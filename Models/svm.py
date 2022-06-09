from dataReader.dataReader import Reader, get_train_test_split
import sklearn
import pandas as pd
from os import getcwd


class SVM():
    def __init__(self, path):
        self.datareader = Reader(path)

        self.path = path

        self.train_dataset = []
        self.test_dataset = []
        self.prep_data()
        self.target = (0, 1, 2)  # 0 is not a major claim, 1 is a major claim
        self.is_trained = False

    def prep_data(self):
        """PReps the training a test data"""
        train_test_split = get_train_test_split(self.path)
        print(train_test_split)
        data = self.datareader.load_from_directory()
        print(data.columns.values)
        train_list = [i for i,(_,y) in enumerate(train_test_split) if y == 0]
        test_list = [i for i,(_,y) in enumerate(train_test_split) if y == 1]
        self.train_dataset = data.loc[data['fileid'].isin(train_list)]
        self.test_dataset = data.loc[data['fileid'].isin(test_list)]


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

        dataset[dataset['label']==-1.0] = 1.0
        dataset[dataset['label']==-2.0] = 2.0

def run():
    path = getcwd()
    path += "/ArgumentAnnotatedEssays-2.0/"
    print(path)
    ml_model = SVM(path)

    ml_model.train()


# print(arg1 + "\n" + arg2)

if __name__ == '__main__':
    pass
