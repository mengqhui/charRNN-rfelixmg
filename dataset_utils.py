import numpy as np
import random

class Text:

    def __init__(self):

        pass

    def load_char_level(self, path, verbose=False):

        text = open(path).read().lower()

        chars = sorted(list(set(text)))

        char_idx = dict((c, i) for i, c in enumerate(chars))
        ixd_char = dict((i, c) for i, c in enumerate(chars))

        if verbose:
            print('Total corpus length:', len(text))
            print('total chars:', len(chars))

        return text, char_idx, ixd_char

    def parse_dataset(self, text, parcel_size=40, step=3, output_level=False, verbose=False):

        if not output_level:
            output_level = parcel_size

        j = len(text) - parcel_size

        x_set = []
        y_set = []
        for i in range(0, j, step):
            x_set.append(text[i: i + parcel_size])
            y_set.append(text[(i + output_level):(i + 1 + parcel_size)])

            if verbose and (i %100 == 0):
                print "[x: '%s' - y: '%s']" % (x_set[-1], y_set[-1])

        if verbose:
            print 'Number of sequences:', len(x_set)


        return x_set, y_set

    def vectorization(self, v_set, char_idx):

        #X = np.zeros((n_samples, pattern_lenght, vocabulary_size))
        V = np.zeros((len(v_set), len(v_set[0]), len(char_idx)), dtype=np.bool)

        for i, x in enumerate(v_set):
            for t, char in enumerate(x):
                V[i, t, char_idx[char]] = 1

        return V

    def split_dataset(self, number_samples, test_percentage):

        indices = range(number_samples)
        random.shuffle(indices)
        tsh_id = number_samples - int(number_samples * test_percentage)
        train_index = np.array(indices[:tsh_id])
        test_index = np.array(indices[tsh_id:])

        return train_index, test_index

    def save_split(self, split_file_name, split_dataset):
        np.save(split_file_name, split_dataset)

    def load_split(self, split_file_name):
        return np.load(split_file_name)









