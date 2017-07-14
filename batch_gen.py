#-*- coding: utf-8 -*-
import numpy as np

def get_batches(raw_data, raw_label, batch_size = 100, shuffle = True):
    """
    """
    data, label = np.array(raw_data), np.array(raw_label)
    data_size = len(data)

    # shuffle data
    if shuffle:
        shuffle_indices = np.random.permutation(range(data_size))
        data = data[shuffle_indices]
        label = label[shuffle_indices]

    # batches generator
    n_batches = data_size // batch_size + 1
    for index in xrange(n_batches):
        start_index = index * batch_size
        end_index = min(data_size, index * batch_size + batch_size)
        yield data[start_index:end_index], label[start_index:end_index]
