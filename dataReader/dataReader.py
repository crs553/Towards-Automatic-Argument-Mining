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
        self.filenames = [i for i in glob(self.__path + "/brat-project-final/*.xml")]
        self.directory = self.filenames[0][:-16]
        print(f"Detected {len(self.filenames)} files:")
        print(self.filenames[0])
        self.filenames.sort()
        for num, i in enumerate(self.filenames):
            t = "\n" if num % 10 == 0 else ""
            print(f"{i[-16:]}\t", end=t)
        print()

    @property
    def path(self) -> str:
        return self.__path

    @property
    def len_dataset(self) -> int:
        return len(self.filenames)

    def setup_train_test(self) -> list:
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

    def load_from_directory(self, rst_files=False, ADU=False):
        """Load all files in the directory, creates relation matrix for them
        Input:
            directory: directory with annotation files
            rst_file: True, if the directory stores RST files as well
            ADU: True for proposition type data loading
        Output:
            dataFrame: pandas DataFrame with samples as rows"""

        data_list = list()

        for (e, annotation_file) in enumerate(self.filenames):
            if annotation_file[-7:] not in ['ann.xml']:
                continue
            if not ADU:
                file_data = self.load_single_file(e, rst_files)
            else:
                file_data = self.load_for_ADU_types(e)
            data_list = data_list + file_data
        dataFrame = pd.DataFrame.from_dict(data_list, orient='columns')
        print('Loaded data length: ' + str(len(dataFrame)))
        return dataFrame

    def load_single_file(self, fileID, rst=False) -> dict:
        """
        Load a single file, creates relation matrix
        :param fileID: index file
        :param rst: True if RST files are stored and used
        :return file_data: dictionary with following features:
                arg1, arg2, argumentationID, label,
                originalArg1, originalArg2, fullText1,
                rstCon, rstConParent - only if RST active,
                positionDiff, positArg1, positArg2,
                sentenceDiff, sen1, sen2 - only if full text exists
        """

        file_data = []
        relations = {}

        data = self.__get_single_data(fileID)

        xmlData = xmltodict.parse(data)

        # if rst_files:
        #     (recovered_string, prop_edu_dict) = load_merge(file_path)
        #     edges = load_brackets(file_path)

        argID = fileID

        matrixLength = len(xmlData['Annotation']['Proposition'])
        relationCount = 0
        totalRelation = matrixLength * matrixLength
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
                        if propositions[i]['TextPosition']['@start'] \
                                != '-1' or propositions[j]['TextPosition'
                        ]['@start'] != '-1':

                            if propositions[i]['TextPosition']['@start'] \
                                    != '-1':
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
                        'fullText1': original_text2,
                    }

                    # if rst_files:
                    #     arg1_range = get_edus(propositions[i]['text'],
                    #                           recovered_string, prop_edu_dict)
                    #     arg2_range = get_edus(propositions[j]['text'],
                    #                           recovered_string, prop_edu_dict)
                    #     arg1_rsts = get_rst_stats(arg1_range, edges)
                    #     arg2_rsts = get_rst_stats(arg2_range, edges)
                    #     cn1 = arg1_rsts['connected_nodes']
                    #     cn2 = arg2_rsts['connected_nodes']
                    #     conn = False
                    #     conn_parent = any([z in cn1 for z in cn2])
                    #     for c in cn1:
                    #         if c in arg2_range:
                    #             conn = True
                    #     for c in cn2:
                    #         if c in arg1_range:
                    #             conn = True
                    #     line_data['rstCon'] = (1 if conn else 0)
                    #     line_data['rstConParent'] = \
                    #         (1 if conn_parent else 0)
                    #
                    # #                    line_data['posEduArg1'] = arg1_range[0]
                    # #                    line_data['posEduArg2'] = arg2_range[0]

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

    def load_for_ADU_types(self, fileID):
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
        data = self.__get_single_data(fileID)

        argID = fileID
        xmlData = xmltodict.parse(data)

        matrixLength = len(xmlData['Annotation']['Proposition'])
        relationCount = 0
        totalRelation = matrixLength * matrixLength
        relationMatrix = (matrixLength, matrixLength)
        relationMatrix = np.zeros(relationMatrix)
        original_text2 = " "

        xmlData = xmltodict.parse(data)

        propositions = xmlData['Annotation']['Proposition']
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
            }
            file_data.append(line_data)
        return file_data

    def fit_tokenize_length_threshold(self,proposition, min_len=1, max_len=30):
        """Drop out too long tokens"""

        if len(sent_tokenize(proposition)) > min_len:
            return True
        elif len(word_tokenize(proposition)) > max_len:
            return True
        else:
            return False

    def __get_single_data(self, fileID):
        with open(self.filenames[fileID], 'r') as file:
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

    print()
    t = reader.load_from_directory()
    print()
    print(t)


    # for a,b in splits:
    #     print(f"{a},{b}")
    # print(splits[401])
    # print(len(splits))
