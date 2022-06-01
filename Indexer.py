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


# Class for storing data about documents
class data_point:
    def __init__(self, name, url, categories, developer, publisher, terms):
        self.name = name
        self.url = url
        self.categories = categories
        self.developer = developer
        self.publisher = publisher
        self.terms = terms

    def get_data_point_by_category(self, cat):
        if cat in self.categories:
            return self

    def get_url_by_name(self, name):
        if name == self.name:
            return self.url

    def get_data_by_name(self, name):
        if name == self.name:
            return self


# class for storing data abotu the elements of the inverted index
class inverted_index:
    def __init__(self, word, documentfrequency, total_frequency, listvals):
        self.word = word
        self.documentfrequency = documentfrequency
        self.total_frequency = total_frequency
        self.listvals = listvals

    def printwords(self):
        print("Word :", self.word, "|| Document Frequency :", self.documentfrequency, " || total frequency",
              self.total_frequency, " || document lists", self.listvals)


# Cleaning the data to check for duplicates and NaN values or empty elements, which can lead to problems in finding unique terms
def clean_steam():
    # Getting data from the csv files using pandas
    data = pd.read_csv(r'BaseData/steam_data.csv')
    data_des = pd.read_csv(r'BaseData/text_content.csv')

    # cleaning all data, removing duplicates and incomplete rows
    data.drop_duplicates(subset='url', keep='last', inplace=True)
    data_des.drop_duplicates(subset='url', keep='last', inplace=True)
    data_des.drop_duplicates(subset='desc', keep='last', inplace=True)

    # removing rows that have null elements in identifying columns
    data = data.dropna(subset=['name'])
    data_des = data_des.dropna(subset=['full_desc'])

    # create dataframes to merge
    url_frame_left = pd.DataFrame(data=data, columns=['url', 'name', 'categories', 'developer', 'publisher'])
    url_frame_right = pd.DataFrame(data=data_des, columns=['url', 'desc'])
    url_frame = pd.merge(url_frame_left, url_frame_right, how='inner', on='url')
    url_frame.drop_duplicates(['url'], keep='last', inplace=True)
    filename = 'Stored_Data/clean_frame'
    outfile = open(filename, 'wb')
    pickle.dump(url_frame, outfile)
    outfile.close()


# Transforming the cleaned data into terms and documents
def terms_stored_frame():
    # opening the cleaned data

    infile = open('Stored_Data/clean_frame', 'rb')
    stored_frame = pickle.load(infile)
    datalist = []

    # get stop words and remove it from the list of terms, using porter stemmer to find lemmas of the words
    stop_words = nltk.corpus.stopwords.words('english')
    stemmer = PorterStemmer()
    all_terms = []

    # iterating through the pandas data frame
    for index, row in stored_frame.iterrows():
        row_terms = ""
        row_terms = " " + str(row['name']) + " " + str(row['categories']) + " " + str(row['developer']) + " " + str(
            row['desc'])
        word1 = word_tokenize(row_terms)
        word1 = [word for word in word1 if word not in stop_words]
        word1 = [word.lower() for word in word1]
        word1 = [stemmer.stem(word) for word in word1]
        all_terms.extend(word1)

        # Get document and corresponding terms, this is useful later
        datalist.append(data_point(str(row['name']), str(row['url']), str(row['categories']), str(row['developer']),
                                   str(row['publisher']), word1))

    # finding unique terms and storing them in a list
    termset = list(dict.fromkeys(all_terms).keys())

    # debugging statement
    print("set of words are : \n", termset, "\n Total terms in dictionary are : ", len(termset))

    # Storing a tuple consisting of all the data and the set of terms
    storage_tuple = (termset, datalist)
    filename = 'Stored_Data/terms_data'
    outfile = open(filename, 'wb')
    pickle.dump(storage_tuple, outfile)
    outfile.close()


def make_inverted_index(train):
    # Calculating the inverted index here for term->documents

    if train:
        inverted_word_dict = {}
        infile = open('Stored_Data/terms_data', 'rb')

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
        filename = 'Stored_Data/inverted_list'
        outfile = open(filename, 'wb')
        pickle.dump(inverted_word_dict, outfile)
        outfile.close()


def make_document_matrix(train):
    if train:
        # opening all the required files

        infile = open('Stored_Data/inverted_list', 'rb')
        inverted_list = pickle.load(infile)
        infile1 = open('Stored_Data/terms_data', 'rb')
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
        filename = 'Stored_Data/document_term_matrix'
        sparse_doc_matrix = csr_matrix(document_term_matrix)
        print(sparse_doc_matrix)
        scipy.sparse.save_npz(filename, sparse_doc_matrix)
        filename_idf = 'Stored_Data/idf_list'
        outfile = open(filename_idf, 'wb')
        pickle.dump(idf_list, outfile)
        outfile.close()


'''
________________________________________________________________________________________________________________________
this function is not being used in the current iteration of the project due to the time it takes ( avg 30 seconds)'''


def process_query(input_query,
                  train,
                  path_to_idf='Stored_Data/idf_list',
                  path_to_terms_dta='Stored_Data/terms_data',
                  path_to_document_lengths='Stored_Data/document_lengths',
                  path_to_doc_mat='Stored_Data/document_term_matrix.npz'):
    # Fro time logging
    start = time.perf_counter()

    # Parsing of the input query
    stop_words = nltk.corpus.stopwords.words('english')
    stemmer = PorterStemmer()
    word1 = word_tokenize(input_query)
    word1 = [word for word in word1 if word not in stop_words]
    print("Query words are : ", word1)
    wordlist = []
    for word in word1:
        syns = wordnet.synsets(word)
        if len(syns) > 0:
            if len(syns[0].lemmas()) > 1:
                print("Syns is : ", syns[0].lemmas()[1].name())
                if syns[0].lemmas()[1].name() not in word1:
                    wordlist.append((syns[0].lemmas()[1].name()))
    word1.extend(wordlist)
    word1 = [stemmer.stem(word) for word in word1]
    word1 = [word.lower() for word in word1]
    print("query words are : \n", word1)
    if train:
        infile = open(path_to_idf, 'rb')
        infile1 = open(path_to_terms_dta, 'rb')
        infile2 = open(path_to_document_lengths, 'rb')
        idf = pickle.load(infile)
        doc_lengths = pickle.load(infile2)
        terms, data_points = pickle.load(infile1)
        terms = list(terms)
        query_row = np.zeros(shape=(1, len(terms)), dtype=np.float32)
        doc_mat = scipy.sparse.load_npz(path_to_doc_mat)
        doc_mat.todense()
        doc_mat = doc_mat.toarray()
        for word in word1:
            if word in terms:
                query_row[0][terms.index(word)] += 1
        print(query_row)
        for elem in range(len(query_row[0])):
            # tf x idf step for the query to find the weights
            query_row[0][elem] = query_row[0][elem] * idf[elem]
        query_length = find_Length_vector_for_query(query_row)
        cosine_sim = {}
        # finding cosine similarities between each document and query vector
        for row in range(len(doc_mat)):
            dot = np.dot(doc_mat[row], query_row[0])
            sim_val = dot / (doc_lengths[row] * query_length)
            cosine_sim[row] = sim_val
        everything = sorted(cosine_sim.items(), key=lambda a: a[1], reverse=True)
        if len(everything) >= 10:
            reverse = everything[:10]
        else:
            reverse = everything
        for element in reverse:
            print(element)
        result_list = []
        for k, v in reverse:
            print(data_points[k].url)
            result_list.append(data_points[k].url)
        end = time.perf_counter()
        diff = end - start
        print("Time taken = ", diff)
        return result_list


'''
________________________________________________________________________________________________________________________
'''


# function to process query based on given inputs
def process_query_reduced(input_query,
                          train,
                          path_to_idf='Stored_Data/idf_list',
                          path_to_terms_dta='Stored_Data/terms_data',
                          path_to_document_lengths='Stored_Data/document_lengths',
                          path_to_doc_mat='Stored_Data/document_term_matrix.npz',
                          path_to_inverted_list='Stored_Data/inverted_list'):
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
    for word in word1:
        syns = wordnet.synsets(word)
        if len(syns) > 0:
            if len(syns[0].lemmas()) > 1:
                print("Syns is : ", syns[0].lemmas()[1].name())
                if syns[0].lemmas()[1].name() not in word1:
                    wordlist.append((syns[0].lemmas()[1].name()))

    # including the synonyms to the actual list of words
    word1.extend(wordlist)

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

        # loading the data in the files to required variables
        idf = pickle.load(infile)
        doc_lengths = pickle.load(infile2)
        terms, data_points = pickle.load(infile1)
        terms = list(terms)
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
            reverse = everything[:10]
        else:
            # Getting the resulting set, if results size is < 10
            reverse = everything

        # # debugging statement meant to check the values
        # for element in reverse:
        #     print(element)

        result_list = []
        for k, v in reverse:
            # Appending urls to the result to be returned
            result_list.append(data_points[k].url)
            print(data_points[k].url)

        # logging time for conclusion of search and indexing
        end = time.perf_counter()
        diff = end - start

        # Time taken for the search to conclude
        print("Time taken = ", diff)

        # returned results
        return result_list


def find_docu_length(train):
    document_lengths = []
    if train:

        # get the document-term matrix and its idf values
        infile = open('Stored_Data/idf_list', 'rb')
        doc_mat = scipy.sparse.load_npz('Stored_Data/document_term_matrix.npz')
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
    filename = 'Stored_Data/document_lengths'
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


if __name__ == "__main__":
    # clean_steam()
    # terms_stored_frame()
    # make_inverted_index(True)
    # make_document_matrix(True)
    # find_docu_length(True)

    # Debugging queries
    process_query_reduced('Earth moves under my feet', True)
