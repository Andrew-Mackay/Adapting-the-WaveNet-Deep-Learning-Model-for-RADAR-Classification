import sys
from six.moves import cPickle
import os
import numpy as np
import pickle


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


def reshape(data):
    return data.reshape(data.shape[0], 1, 75, 75)


def reshape_labels(labels):
    return np.reshape(labels, (len(labels), 1))


def load_dataset(path):
    file = open(path + "test_data.pkl", 'rb')
    test_data = pickle.load(file)
    file.close()
    test_data = reshape(test_data)
    file = open(path + "train_data.pkl", 'rb')
    train_data = pickle.load(file)
    file.close()
    train_data = reshape(train_data)

    file = open(path + "test_labels.pkl", 'rb')
    test_labels = pickle.load(file)
    file.close()
    test_labels = reshape_labels(test_labels)
    file = open(path + "train_labels.pkl", 'rb')
    train_labels = pickle.load(file)
    file.close()
    train_labels = reshape_labels(train_labels)
    return {
        "test_data": test_data,
        "train_data": train_data,
        "test_labels": test_labels,
        "train_labels": train_labels
    }


def load_data_specific_files(name_path_dict):
    '''
    Dictionary has form:
     {name:{
        "path":"/path/to/file",
        "type":"label" or "data"
       }
    '''
    datasets = {}
    for name, path in name_path_dict.items():
        # split by "/" to get name of file, "." to remove extension then "_" to get type
        object_type = path.split("/")[-1].split(".")[0].split("_")[1]
        file = open(path, 'rb')
        unpickled = pickle.load(file)
        file.close()
        if object_type == "labels":
            datasets[name] = reshape_labels(unpickled)
        elif object_type == "data":
            datasets[name] = reshape(unpickled)

    return datasets
