import sys
from six.moves import cPickle
import os
import numpy as np


def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding="bytes")
        # decode utf8
        for k, v in d.items():
            # added check as otherwise tries to decode a string
            if type(k) is not str:
                del(d[k])
                d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 1, 75, 75)
    return data, labels


def load_data(path, nb_train_samples):
    X_train = np.zeros((nb_train_samples, 1, 75, 75), dtype="uint8")
    y_train = np.zeros((nb_train_samples,), dtype="uint8")
    offset = int(nb_train_samples/5)
    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        X_train[(i-1)*offset:i*offset, :, :, :] = data
        y_train[(i-1)*offset:i*offset] = labels

    fpath = os.path.join(path, 'test_batch')
    X_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    return (X_train, y_train), (X_test, y_test)
