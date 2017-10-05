# Team - hw2group 428
# Nguyen Minh Thong Huynh (z5170141)
# Payal Bawa (z5132512)


import tensorflow as tf
import numpy as np
import collections

data_index = 0

def generate_batch(data, batch_size, skip_window):

    """
    Generates a mini-batch of training data for the training CBOW
    embedding model.
    :param data (numpy.ndarray(dtype=int, shape=(corpus_size,)): holds the
        training corpus, with words encoded as an integer
    :param batch_size (int): size of the batch to generate
    :param skip_window (int): number of words to both left and right that form
        the context window for the target word.
    Batch is a vector of shape (batch_size, 2*skip_window), with each entry for the batch containing all the context words, with the corresponding label being the word in the middle of the context
    """

    global data_index

    #Conditions
    #assert batch_size % num_skips == 0
    #assert num_skips <= 2 * skip_window

    # [ skip_window target skip_window ]
    span_size = 2 * skip_window + 1

    batch = np.ndarray(shape=(batch_size,span_size-1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # create buffer to store span window for CBOW
    word_buffer = collections.deque(maxlen=span_size)

    if data_index + span_size > len(data):
        # go back to the start if reaching the end of data
        data_index = 0
        word_buffer.extend(data[data_index:data_index + span_size])
    data_index += span_size
    for i in range(batch_size):
        target = skip_window  # target label at the center of the buffer

        word_pos = 0
        for j in range(span_size):
            if j==span_size//2: # skip middle word which is target word
                continue
            batch[i, word_pos] = word_buffer[j]
            word_pos += 1
        labels[i, 0] = word_buffer[target]

        word_buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels

def get_mean_context_embeds(embeddings, train_inputs):
    """
    :param embeddings (tf.Variable(shape=(vocabulary_size, embedding_size))
    :param train_inputs (tf.placeholder(shape=(batch_size, 2*skip_window))
    returns:
        `mean_context_embeds`: the mean of the embeddings for all context words
        for each entry in the batch, should have shape (batch_size,
        embedding_size)
    """
    # cpu is recommended to avoid out of memory errors, if you don't
    # have a high capacity GPU
    with tf.device('/cpu:0'):
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        mean_context_embeds = tf.reduce_sum(embed, 1)
    return mean_context_embeds
