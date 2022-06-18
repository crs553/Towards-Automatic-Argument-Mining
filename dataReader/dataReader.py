from glob import glob

import numpy as np
import pandas as pd
import tqdm
import xmltodict
from nltk.tokenize import sent_tokenize, word_tokenize


class Reader:
    def __init__(self, __path: str) -> None:
        """
        Initialisation of the reader
        :param __path: the path to the folder containing the unzipped files
        """

        self.__path = __path
        print("Loading files")
        self.filenames = [i for i in glob(self.__path + "/brat-project-final/*.xml")]
        self.directory = self.filenames[0][:-16]
        print(f"Detected {len(self.filenames)} files:")
        self.filenames.sort()
        # for num, i in enumerate(self.filenames):
        #     _end = "\n" if num % 12 == 0 and num != 0 else ""
        #     print(f"{i[-16:]}\t", end=_end)
        # print()

    @property
    def path(self) -> str:
        return self.__path

    @property
    def len_dataset(self) -> int:
        return len(self.filenames)

    # def setup_train_test(self) -> list:
    #     """
    #     Finds train and test type for each file splits and returns them as list
    #     :return:
    #     """
    #     f = open(self.__path + "/train-test-split.csv", "r")
    #     file = f.read()
    #     f.close()
    #     clean_file = file.split("\n")
    #     clean_file = clean_file[1:-1]
    #     splits = []
    #     for f_and_t in clean_file:
    #         a = f_and_t[1:-1].split(";")
    #         splits.append((a[0][:-1], a[1][1:]))
    #
    #     split_new = []
    #     for i, j in splits:
    #         z = 1
    #         if j == "TRAIN":
    #             z = 0
    #
    #         pos = int(i[-3:])
    #         split_new.append((pos, z))
    #
    #     return split_new

    def load_from_directory(self, adu=False):
        """Load all files in the directory, creates relation matrix for them
        Input:
            directory: directory with annotation files
            rst_file: True, if the directory stores RST files as well
            ADU: True for proposition type data loading
        Output:
            dataFrame: pandas DataFrame with samples as rows"""

        data_list = list()
        pdbar = tqdm.tqdm(total=len(self.filenames))
        for (e, annotation_file) in enumerate(self.filenames):
            if annotation_file[-7:] not in ['ann.xml']:
                continue
            if not adu:
                file_data = self.load_single_file(e)
            else:
                file_data = self.load_for_adu_types(e)
            data_list = data_list + file_data
            pdbar.update(1)
        dataFrame = pd.DataFrame.from_dict(data_list, orient='columns')
        print('Loaded data length: ' + str(len(dataFrame)))
        return dataFrame

    def load_single_file(self, file_id) -> list:
        """
        Load a single file, creates relation matrix
        :param file_id: index file
        :return file_data: list containing dictionary with following features:
                arg1, arg2, argumentationID, label,
                originalArg1, originalArg2, fullText1,
                rstCon, rstConParent - only if RST active,
                positionDiff, positArg1, positArg2,
                sentenceDiff, sen1, sen2 - only if full text exists
        """

        file_data = []

        data = self.__get_single_data(file_id)

        xmlData = xmltodict.parse(data)

        argID = file_id

        matrixLength = len(xmlData['Annotation']['Proposition'])
        relationMatrix = (matrixLength, matrixLength)
        relationMatrix = np.zeros(relationMatrix)
        sent_tokenize_list = None

        original_text = None
        original_text2 = None
        sens = None

        propositions = xmlData['Annotation']['Proposition']
        if 'OriginalText' in xmlData['Annotation']:
            original_text = xmlData['Annotation']['OriginalText']
            original_text2 = original_text.replace('\n', ' ')
            sent_tokenize_list = sent_tokenize(original_text)
            sens = len(sent_tokenize_list)

        for prop_id in range(len(propositions)):
            currentProposition = propositions[prop_id]

            if currentProposition['ADU']['@type'] != 'conclusion' and 'Relation' in currentProposition.keys():

                partners = list()
                relationTypeList = list()

                if currentProposition['Relation'].__class__ == list().__class__:
                    for relation in range(len(currentProposition['Relation'])):
                        relation_data = currentProposition['Relation'][relation]

                        partners.append(relation_data['@partnerID'])
                        relationTypeList.append(relation_data['@typeBinary'])
                else:

                    relation_data = currentProposition['Relation']
                    partners.append(relation_data['@partnerID'])
                    relationTypeList.append(relation_data['@typeBinary'])

                for partner_id in range(len(partners)):
                    for prop_id2 in range(len(propositions)):
                        if partners[partner_id] == propositions[prop_id2]['@id']:
                            if relationTypeList[partner_id] == '0':
                                relationMatrix[prop_id][prop_id2] = 1
                                relationMatrix[prop_id2][prop_id] = -1
                            elif relationTypeList[partner_id] == '1':

                                relationMatrix[prop_id][prop_id2] = 2
                                relationMatrix[prop_id2][prop_id] = -2
                            else:
                                relationMatrix[prop_id][prop_id2] = -3

        sen1 = None
        sen2 = None
        for i in range(len(relationMatrix)):
            for j in range(len(relationMatrix[i])):
                if i != j and relationMatrix[i][j] > -3:
                    proposition1 = propositions[i]['text']
                    proposition2 = propositions[j]['text']
                    if self.fit_tokenize_length_threshold(proposition1) \
                            or self.fit_tokenize_length_threshold(proposition2):
                        continue

                    originalSentenceArg1 = propositions[i]['text']
                    originalSentenceArg2 = propositions[j]['text']

                    if 'TextPosition' in propositions[i].keys():
                        if propositions[i]['TextPosition']['@start'] != '-1' \
                                or propositions[j]['TextPosition']['@start'] != '-1':

                            if propositions[i]['TextPosition']['@start'] != '-1':
                                for sentence in sent_tokenize_list:

                                    if propositions[i]['text'] in sentence:
                                        originalSentenceArg1 = sentence
                                        sen1 = sent_tokenize_list.index(sentence)

                            if propositions[j]['TextPosition']['@start'] != '-1':

                                for sentence in sent_tokenize_list:
                                    if propositions[j]['text'] in sentence:
                                        originalSentenceArg2 = sentence
                                        sen2 = sent_tokenize_list.index(sentence)

                    line_data = {
                        'argumentationID': argID,
                        'arg1': propositions[i]['text'],
                        'originalArg1': originalSentenceArg1,
                        'arg2': propositions[j]['text'],
                        'originalArg2': originalSentenceArg2,
                        'label': relationMatrix[i][j],
                        'originalLabel': relationMatrix[i][j],
                        'fullText1': original_text2,
                        'fileid': file_id,
                    }

                    positArg1 = int(propositions[i]['TextPosition']['@start'])
                    positArg2 = int(propositions[j]['TextPosition']['@start'])
                    if positArg1 != -1 and positArg2 != -1:
                        posit = abs((positArg1 - positArg2) / len(original_text))
                        line_data['positionDiff'] = posit
                        line_data['positArg1'] = positArg1 / len(original_text)
                        line_data['positArg2'] = positArg2 / len(original_text)
                        senit = abs(sen1 - sen2)
                        line_data['sentenceDiff'] = senit / sens
                        line_data['sen1'] = sen1 / sens
                        line_data['sen2'] = sen2 / sens

                    file_data.append(line_data)
        return file_data

    def load_for_adu_types(self, __file_id):
        """Loads ADU type features.
        Input:
            fileID - index for the processed files
        Output:
            file_data: dictionary with the features stored:
                       arg1, argumentationID, label,
                       originalArg1, fullText1, positArg1
        """
        file_data = list()
        relationMatrix = {}
        data = self.__get_single_data(__file_id)

        argID = __file_id
        xmlData = xmltodict.parse(data)

        matrixLength = len(xmlData['Annotation']['Proposition'])
        relationCount = 0
        totalRelation = matrixLength * matrixLength
        relationMatrix = (matrixLength, matrixLength)
        relationMatrix = np.zeros(relationMatrix)
        original_text2 = " "

        xmlData = xmltodict.parse(data)

        propositions = xmlData['Annotation']['Proposition']
        sent_tokenize_list = None
        if 'OriginalText' in xmlData['Annotation']:
            original_text = xmlData['Annotation']['OriginalText']
            original_text2 = original_text.replace('\n', ' ')
            sent_tokenize_list = sent_tokenize(original_text)
            sens = len(sent_tokenize_list)

        for prop_id in range(len(propositions)):
            currentProposition = propositions[prop_id]

            if currentProposition['ADU']['@type'] == 'conclusion':
                aduType = 2
            elif currentProposition['ADU']['@type'] == 'claim':
                aduType = 1
            elif currentProposition['ADU']['@type'] == 'premise':
                aduType = 0
            else:
                err_ADU = currentProposition['ADU']['@type']
                raise ValueError('Unexpected ADU type: ' + err_ADU)

            arg1 = currentProposition['text']
            originalSentenceArg1 = arg1
            positArg1 = -1

            if currentProposition['TextPosition']['@start'] != '-1':
                for sentence in sent_tokenize_list:

                    if arg1 in sentence:
                        originalSentenceArg1 = sentence
                        sen1 = sent_tokenize_list.index(sentence)

                positArg1 = int(currentProposition['TextPosition']['@start'])
            line_data = {
                'argumentationID': argID,
                'arg1': arg1,
                'originalArg1': originalSentenceArg1,
                'label': aduType,
                'fullText1': original_text2,
                'positArg1': positArg1 / len(original_text2),
                'fileid': __file_id
            }
            file_data.append(line_data)
        return file_data

    @staticmethod
    def fit_tokenize_length_threshold(proposition, min_len=1, max_len=30):
        """Drop out too long tokens"""

        if len(sent_tokenize(proposition)) > min_len:
            return True
        elif len(word_tokenize(proposition)) > max_len:
            return True
        else:
            return False

    def __get_single_data(self, __file_id):
        with open(self.filenames[__file_id], 'r') as file:
            data = file.read()

        return data

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

        # total number of possible relations
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
                'fileid': fileid
            }
            file_data.append(line_data)
        return file_data


def get_train_test_split(path: str) -> list:
    f = open(path + "train-test-split.csv", "r")
    if f is None:
        raise FileNotFoundError(f"Could not find file at {path}train-test-split.csv")
    file = f.read()
    file = file.split("\n")
    format_file = []
    for x in file[1:]:
        label = 0
        if "test" in x.lower():
            label = 1
        format_file.append((int(x[6:9]), label))
    return format_file

