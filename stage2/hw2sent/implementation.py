# Team - hw2group 428
# Nguyen Minh Thong Huynh (z5170141)
# Payal Bawa (z5132512)

import tensorflow as tf
import numpy as np
import glob  # this will be useful when reading reviews from file
import os
import tarfile

batch_size = 50
numClasses = 2
maxSeqLength = 40
numDimensions = 50
lstmUnits = 75

import re
import string


import sys


def check_file(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        print("please make sure {0} exists in the current directory".format(filename))
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            "File {0} didn't have the expected size. Please ensure you have downloaded the assignment files correctly".format(
                filename))
    return filename


# Read the data into a list of strings.
def extract_data(filename):
    """Extract data from tarball and store as list of strings"""
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data2/')):
        with tarfile.open(filename, "r") as tarball:
            dir = os.path.dirname(__file__)
            tarball.extractall(os.path.join(dir, 'data2/'))
    return


def read_data_to_array_words():
    if os.path.exists(os.path.join(os.path.dirname(__file__), "reviews.npy")):
        print("loading saved parsed reviews, to reparse, delete 'reviews.npy'")
        reviews = np.load("reviews.npy")
    else:
        print("READING DATA")

        dir = os.path.dirname(__file__)
        file_list_positives = glob.glob(os.path.join(dir,
                                                     'data2/pos/*'))
        file_list_negatives = glob.glob(os.path.join(dir,
                                                     'data2/neg/*'))

        reviews = []
        # strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        print("Parsing %s positives files" % len(file_list_positives))
        for f in file_list_positives:
            with open(f, "r", encoding="utf-8") as openf:
                s = openf.read()
                #no_punct = ''.join(c for c in s if c not in string.punctuation)
                reviews.append(s.split())

        print("Parsing %s negatives files" % len(file_list_positives))
        for f in file_list_negatives:
            with open(f, "r", encoding="utf-8") as openf:
                s = openf.read()
                #no_punct = ''.join(c for c in s if c not in string.punctuation)
                reviews.append(s.split())

        np.save("reviews", reviews)
    return reviews


def valid_word(word):
    if word in ['a', 'an', 'the', 'actor', 'actress', 'movie', 'cast', 'story', 'plot', 'director', 'film', 'all'
        ,'and', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'between', 'both', 'by', 'could',
                'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'having', 'he', 'hed',
                'hell', 'hes', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how',
                'i', 'id', 'ill', 'im', 'ive', 'if', 'in', 'into', 'is', 'it', 'its', 'its', 'itself', 'lets', 'me',
                'my', 'has', 'will', 'myself', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
                'ourselves', 'out', 'own', 'she', 'shed', 'shell', 'shes', 'so', 'some', 'such', 'than', 'that',
                'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres', 'these', 'they',
                'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', 'until', 'up', 'was',
                'we', 'wed', 'weve', 'were', 'what', 'whats', 'when', 'whens', 'where', 'wheres', 'which', 'while', 'who',
                'whos', 'whom', 'with', 'would', 'you', 'youd', 'youll', 'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves']:
        return False
    else:
        return True


def clear_format(string):
    return clean_tag_and_char(string).lower()


def clean_tag_and_char(string):
    re_rule = re.compile('<.*?>')
    clean_tag = re.sub(re_rule, '', string)
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    clean_char = re.sub(strip_special_chars, '', clean_tag)
    return clean_char


def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form
    """
    if os.path.exists(os.path.join(os.path.dirname(__file__), "data.npy")):
        print("loading saved parsed vector data, to reparse, delete 'data.npy'")
        data = np.load("data.npy")
    else:
        filename = check_file('reviews.tar.gz', 14839260)
        extract_data(filename)  # unzip
        array_words = read_data_to_array_words()

        data = []
        for row in array_words:
            row_temp=[]
            clear_row = (clear_format(' '.join(row))).split()
            for word in clear_row:

                if valid_word(word):
                    if word in glove_dict:
                        row_temp.append(glove_dict[word])
                    else:
                        row_temp.append(glove_dict['UNK'])

                else:
                    continue

                if len(row_temp) == 40:
                    break

            if len(row_temp) < 40:
                padding = 40 - len(row_temp)
                for i in range(padding):
                    row_temp.append(0)

            data.append(row_temp)
            assert len(row_temp) == 40

        np.save("data", data)

    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    if os.path.exists(os.path.join(os.path.dirname(__file__), "embeddings.npy")):
        print("loading saved parsed embeddings, to reparse, delete 'embeddings.npy'")
        print("loading saved parsed word_index_dict, to reparse, delete 'word_index_dict.npy'")
        embeddings_np = np.load("embeddings.npy")
        word_index_dict = np.load("word_index_dict.npy").item()

    else:

        data = open("glove.6B.50d.txt", 'r', encoding="utf-8")
        # if you are running on the CSE machines, you can load the glove data from here
        # data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")

        embeddings_np = np.zeros([400001, 50], dtype=np.float32)
        word_index_dict = {}
        word_index_dict['UNK'] = 0

        row_pos = 1
        for line in data.readlines():

            row = line.strip().split(' ')
            target_word = row[0]
            vector_temp = row[1:]

            assert len(vector_temp) == 50

            embeddings_np[row_pos] = vector_temp

            word_index_dict[target_word] = row_pos
            row_pos += 1

        np.save("embeddings", embeddings_np)
        np.save("word_index_dict", word_index_dict)

    return embeddings_np, word_index_dict


# -------- test area ---------
# embeddings, word_index_dict = load_glove_embeddings()
#
# # print ('word_index',word_index_dict)
# key, value = word_index_dict.popitem()
# print('key', key)
# # print (value)
# print('embd', embeddings[value])
#
# data = load_data(word_index_dict)
# print('data', data)
# sentence_0 = data[0]
# print("Vector index data:")
# #
# print(sentence_0)
#
# print("Sentence:")
# for index in sentence_0:
#     for key, value in word_index_dict.items():
#         if value == index:
#             print(key)
#
# print("Vector embeddings for each words (only display first 3 vectors):")
# for index in (sentence_0[:20]):
#     print(embeddings[index])


# ---------- test area ends ---------

def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, dropout_keep_prob, optimizer, accuracy and loss
    """
    labels = tf.placeholder(tf.float32, [batch_size, numClasses])
    input_data = tf.placeholder(tf.int32, [batch_size, maxSeqLength])
    # init_state = tf.zeros([batch_size, lstmUnits])
    # data1 = tf.Variable(tf.zeros([batch_size, maxSeqLength, numDimensions]),dtype=tf.float32)

    #embd = tf.Variable(glove_embeddings_arr) ??????????????????????????
    embd = glove_embeddings_arr

    # data = tf.nn.embedding_lookup((tf.convert_to_tensor(glove_embeddings_arr)),input_data)
    data1 = tf.nn.embedding_lookup(embd, input_data)
    # tf.convert_to_tensor(arg, dtype=tf.float32)
    # with tf.variable_scope('basic_lstm'):
    lstmCell = tf.contrib.rnn.LSTMCell(lstmUnits)

    #lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

    # init_state = lstmCell.zero_state(batch_size, tf.float32)
    # with tf.variable_scope('dyn_rnn'):
    value, ops = tf.nn.dynamic_rnn(lstmCell, data1, dtype=tf.float32) ## define backprogation through time 40
    # value, final_state = tf.nn.dynamic_rnn(lstmCell, data1, initial_state=init_state)
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)
    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer(0.0002).minimize(loss)

    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
