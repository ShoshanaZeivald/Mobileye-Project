import pickle


def read_txt(file):
    with open(file, "r") as f:
        lines = [line.rstrip() for line in f]
    return lines


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as pklfile:
        data = pickle.load(pklfile, encoding='latin1')
    return data

