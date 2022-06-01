import math
import os
from math import log
from nltk.corpus import wordnet
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
from collections import Counter
import WebScraper
import pandas as pd
import pickle
import h5py
from scipy.sparse import csr_matrix
import scipy
import time


class data_point:
    def __init__(self, name, terms):
        self.name = name
        self.terms = terms

    def get_data_by_name(self, name):
        if name == self.name:
            return self


class inverted_index:
    def __init__(self, word, documentfrequency, total_frequency, listvals):
        self.word = word
        self.documentfrequency = documentfrequency
        self.total_frequency = total_frequency
        self.listvals = listvals

    def printwords(self):
        print("Word :", self.word, "|| Document Frequency :", self.documentfrequency, " || total frequency",
              self.total_frequency, " || document lists", self.listvals)


def clean_data():
    filepath = 'TestData/cacm.txt'
    mylines = []
    Doc_Data = False
    Doc_Author = False
    Doc_Start = False
    Doc_Name = []
    Author = []
    Data = []
    doc_dict = {}
    with open(filepath, 'rt') as myfile:
        for line in myfile:
            mylines.append(line)
    datalines = ""
    index = 0
    ignored = ['.T', '.W', '.N', '.K', '.B']
    for linenum in range(len(mylines)):

        if not Doc_Start and '.T' in mylines[linenum]:
            Doc_Start = True
            Doc_Data = True
        if Doc_Start:
            if '.X' not in mylines[linenum]:
                if Doc_Data:
                    Doc_Name.append(mylines[linenum + 1])
                    Doc_Data = False
                if mylines[linenum] not in ignored:
                    datalines += mylines[linenum]
            else:
                index += 1
                Data.append(datalines)
                Doc_Start = False
                doc_dict[index] = Data[index - 1]
                datalines = ""

    for i in doc_dict.keys():
        print("Data is : \n", doc_dict[i])
    filename = 'Test_Storage/clean_frame_test'
    outfile = open(filename, 'wb')
    pickle.dump(doc_dict, outfile)
    outfile.close()
    print(len(doc_dict.keys()))
    print("Number of doc names is : ", len(Doc_Name))
    docname_file = 'Test_Storage/doc_names'
    outfile1 = open(docname_file, 'wb')
    pickle.dump(Doc_Name, outfile1)
    outfile1.close()


def terms_stored_frame():
    # opening the cleaned data

    infile = open('Test_Storage/clean_frame_test', 'rb')
    stored_frame = pickle.load(infile)
    datalist = []

    # get stop words and remove it from the list of terms, using porter stemmer to find lemmas of the words
    stop_words = nltk.corpus.stopwords.words('english')
    stemmer = PorterStemmer()
    all_terms = []

    for k, row_terms in stored_frame.items():
        word1 = word_tokenize(row_terms)
        word1 = [word for word in word1 if word not in stop_words]
        word1 = [word.lower() for word in word1]
        word1 = [stemmer.stem(word) for word in word1]
        all_terms.extend(word1)

        # Get document and corresponding terms, this is useful later
        datalist.append(data_point(str(k), word1))

    # finding unique terms and storing them in a list
    termset = list(dict.fromkeys(all_terms).keys())

    # debugging statement
    print("set of words are : \n", termset, "\n Total terms in dictionary are : ", len(termset))

    # Storing a tuple consisting of all the data and the set of terms
    storage_tuple = (termset, datalist)
    filename = 'Test_Storage/terms_data'
    outfile = open(filename, 'wb')
    pickle.dump(storage_tuple, outfile)
    outfile.close()


def make_inverted_index(train):
    # Calculating the inverted index here for term->documents

    if train:
        inverted_word_dict = {}
        infile = open('Test_Storage/terms_data', 'rb')

        # load the required files

        terms, data_points = pickle.load(infile)
        wordindex = 0
        for word in terms:
            wordindex += 1
            inverted_word_dict[word] = inverted_index(word, 0, 0, [])
            print("selecting word number : ", wordindex, "\t with word : ", word)

            # iterating through the data points to find the required terms in the document
            for data_struct in data_points:
                if word in data_struct.terms:
                    inverted_word_dict[word].documentfrequency += 1
                    countword = Counter(data_struct.terms)
                    inverted_word_dict[word].total_frequency += countword[word]
                    print("Document at index : ", data_points.index(data_struct))
                    inverted_word_dict[word].listvals.append((data_points.index(data_struct), countword[word]))

        # storing the inverted index
        filename = 'Test_Storage/inverted_list'
        outfile = open(filename, 'wb')
        pickle.dump(inverted_word_dict, outfile)
        outfile.close()


def make_document_matrix(train):
    if train:
        # opening all the required files

        infile = open('Test_Storage/inverted_list', 'rb')
        inverted_list = pickle.load(infile)
        infile1 = open('Test_Storage/terms_data', 'rb')
        terms, data_points = pickle.load(infile1)
        terms = list(terms)
        idf_list = []
        document_lengths = []
        total_documents = len(data_points)

        # idf calculation part
        for term_index in range(len(terms)):
            doc_freq = inverted_list[terms[term_index]].documentfrequency
            # Calculating idf values here
            idf_term = log((total_documents / doc_freq), 2)
            idf_list.append(idf_term)
        document_term_matrix = np.zeros(shape=(len(data_points), len(terms)), dtype=np.double)
        for term_index in range(len(terms)):
            term_val = terms[term_index]
            inverted_item = inverted_list[term_val].listvals

            # find the document id and the tf from the inverted list of terms
            for doc, freq in inverted_item:
                # tf x idf step to find weights
                document_term_matrix[doc][term_index] = freq * idf_list[term_index]

                # debugging statement
                print("Values for rows and columns are : ", (doc, term_index), "\twith value:\t",
                      document_term_matrix[doc][term_index])

        # saving the data as a scipy sparse matrix, or the size was over 15 GB.
        filename = 'Test_Storage/document_term_matrix'
        sparse_doc_matrix = csr_matrix(document_term_matrix)
        print(sparse_doc_matrix)
        scipy.sparse.save_npz(filename, sparse_doc_matrix)
        filename_idf = 'Test_Storage/idf_list'
        outfile = open(filename_idf, 'wb')
        pickle.dump(idf_list, outfile)
        outfile.close()


def find_docu_length(train):
    document_lengths = []
    if train:

        # get the document-term matrix and its idf values
        infile = open('Test_Storage/idf_list', 'rb')
        doc_mat = scipy.sparse.load_npz('Test_Storage/document_term_matrix.npz')
        doc_mat_dense = csr_matrix.toarray(doc_mat)
        idf = pickle.load(infile)
        for r in range(len(doc_mat_dense)):
            sumsq = 0
            for c in range(len(doc_mat_dense[r])):
                sumsq += (doc_mat_dense[r][c] * doc_mat_dense[r][c])

            # square root of sum of squares is the length of the document vector
            document_lengths.append(math.sqrt(sumsq))
            print("getting the value of sum of squares for document vector", r, " to be : ", math.sqrt(sumsq))
    print(document_lengths)

    # Save the data
    filename = 'Test_Storage/document_lengths'
    outfile = open(filename, 'wb')
    pickle.dump(document_lengths, outfile)
    outfile.close()


# find the length of a vector
def find_Length_vector_for_query(vector):
    for row in range(len(vector)):
        sumsq = 0.
        for col in range(len(vector[row])):
            sumsq += (vector[row][col] * vector[row][col])
            # print("getting the value of sum of squares for document vector", row, " to be : ", math.sqrt(sumsq))
        sumsq = math.sqrt(sumsq)
        return sumsq


def process_query_reduced(input_query,
                          train,
                          path_to_idf='Test_Storage/idf_list',
                          path_to_terms_dta='Test_Storage/terms_data',
                          path_to_document_lengths='Test_Storage/document_lengths',
                          path_to_doc_mat='Test_Storage/document_term_matrix.npz',
                          path_to_inverted_list='Test_Storage/inverted_list',
                          path_to_names='Test_Storage/doc_names'):
    # Fro time logging
    start = time.perf_counter()

    # Parsing of the input query
    stop_words = nltk.corpus.stopwords.words('english')
    stemmer = PorterStemmer()
    word1 = word_tokenize(input_query)
    word1 = [word for word in word1 if word not in stop_words]
    print("Query words are : ", word1)
    wordlist = []

    # Extending query to also include one synonym of the word from wordnet
    # for word in word1:
    #     syns = wordnet.synsets(word)
    #     if len(syns) > 0:
    #         if len(syns[0].lemmas()) > 1:
    #             print("Syns is : ", syns[0].lemmas()[1].name())
    #             if syns[0].lemmas()[1].name() not in word1:
    #                 wordlist.append((syns[0].lemmas()[1].name()))
    #
    # # including the synonyms to the actual list of words
    # word1.extend(wordlist)

    # applying porter stemmer to lemmatize the words
    word1 = [stemmer.stem(word) for word in word1]

    # converting all words to lower case for uniformity
    word1 = [word.lower() for word in word1]

    # debug statement
    print("query words are : \n", word1)
    if train:
        # opening all the files to the corresponding saved data in idf, inverted list and document matrix
        infile = open(path_to_idf, 'rb')
        infile1 = open(path_to_terms_dta, 'rb')
        infile2 = open(path_to_document_lengths, 'rb')
        infile3 = open(path_to_inverted_list, 'rb')
        infile4 = open(path_to_names, 'rb')
        # loading the data in the files to required variables
        idf = pickle.load(infile)
        doc_lengths = pickle.load(infile2)
        terms, data_points = pickle.load(infile1)
        terms = list(terms)
        names = pickle.load(infile4)
        inverted_dict = pickle.load(infile3)
        query_row = np.zeros(shape=(1, len(terms)), dtype=np.float32)
        doc_list = []

        # convert the saved sparse matrix to a dense matrix
        doc_mat = scipy.sparse.load_npz(path_to_doc_mat)
        doc_mat.todense()
        doc_mat = doc_mat.toarray()

        # find reduced documents by ID based on items present
        for word in word1:
            # searching for the word from the query in the inverted term-document dictionary
            if word in inverted_dict.keys():
                inverted_docs = inverted_dict[word].listvals
                # Getting document indicies from the term-document dictionary and saving these in a relevant list
                for doc, freq in inverted_docs:
                    doc_list.append(doc)
        doc_set = set(doc_list)
        doc_list = list(doc_set)
        if len(doc_list) == 0:
            # if no data is present, then return a None result
            print("Nothing is present")
            return []
        for word in word1:
            if word in terms:
                query_row[0][terms.index(word)] += 1
        print(query_row)
        for elem in range(len(query_row[0])):
            # tf x idf step for the query to find the weights
            query_row[0][elem] = query_row[0][elem] * idf[elem]

        # finding the length of the query vector
        query_length = find_Length_vector_for_query(query_row)

        # dictionary to store cosine similarities
        cosine_sim = {}

        # finding cosine similarities between relevant documents and query vector
        for row in doc_list:
            dot = np.dot(doc_mat[row], query_row[0])
            sim_val = dot / (doc_lengths[row] * query_length)
            cosine_sim[row] = sim_val

        # sorting cosine similarities by value of similarity
        everything = sorted(cosine_sim.items(), key=lambda a: a[1], reverse=True)
        if len(everything) >= 10:
            # Getting the top ten elements
            reverse = everything[:5]
        else:
            # Getting the resulting set, if results size is < 10
            reverse = everything

        '''debugging statement meant to check the values'''
        for element in reverse:
            print(element)

        result_list = []
        for k, v in reverse:
            # Appending urls to the result to be returned
            num_name = str(k+1) + " : " + names[k]
            result_list.append(num_name)
        for nameval in result_list:
            print(nameval)

        # logging time for conclusion of search and indexing
        end = time.perf_counter()
        diff = end - start

        # Time taken for the search to conclude
        print("Time taken = ", diff)


def get_all_queries():
    filepath = 'TestData/query.txt'
    mylines = []
    Doc_Data = False
    Doc_Author = False
    Doc_Start = False
    Doc_Name = []
    Author = []
    Query = []
    q_dict = {}
    query_data = ""
    with open(filepath, 'rt') as myfile:
        for line in myfile:
            mylines.append(line)
    index = 1
    for linenum in range(len(mylines)):
        if not Doc_Start and '.I' in mylines[linenum]:
            Doc_Start = True
        elif Doc_Start and '.I' in mylines[linenum]:
            Doc_Start = False
            Query.append(query_data)
            q_dict[index] = Query[index-1]
            index +=1
            query_data = ""
        if '.I' not in mylines[linenum] and Doc_Start:
            query_data += mylines[linenum]

    return Query, q_dict


if __name__ == '__main__':
    clean_data()
    terms_stored_frame()
    make_inverted_index(True)
    make_document_matrix(True)
    find_docu_length(True)
    Queries, q_dict = get_all_queries()
    Query_slice = Queries[:2]
    for q in Queries:
        print(q)
        process_query_reduced(q, True)
