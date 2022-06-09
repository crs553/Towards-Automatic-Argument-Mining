from dataReader.dataReader import Reader, get_train_test_split
import sklearn
from os import getcwd


class SVM():
    def __init__(self, path):
        self.datareader = Reader(path)

        self.path = path

        self.train_dataset = []
        self.test_dataset = []
        self.prep_data()
        self.target = (0, 1)  # 0 is not a major claim, 1 is a major claim
        self.is_trained = False

    def prep_data(self):
        """PReps the training a test data"""
        train_test_split = get_train_test_split(self.path)
        for i, (_, label) in enumerate(train_test_split):
            data = self.datareader.load_single_file(i)
            if label == 0:
                self.train_dataset.append(data)
            else:
                self.test_dataset.append(data)

    def train(self):
        file = self.train_dataset[0]
        print(file)
        self.is_trained = True
        return None


def run():
    path = getcwd()
    path += "/ArgumentAnnotatedEssays-2.0/"
    print(path)
    ml_model = SVM(path)

    ml_model.train()


# print(arg1 + "\n" + arg2)

if __name__ == '__main__':
    pass
