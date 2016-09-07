# http://stackoverflow.com/questions/7427101/dead-simple-argparse-example-wanted-1-argument-3-results
# https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py#L74

import os
from tornado.platform import epoll
import math

from numpy.f2py.auxfuncs import throw_error

from charRNN import charRNN
import sys, argparse, random, time
import numpy as np
from sklearn.cross_validation import KFold
import dataset_utils as d_utils


import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def main():

    try:

        parser = argparse.ArgumentParser(description='Training LSTM to generate text in char-rnn level')
        parser.add_argument('-d', '--dataset', help='Input new dataset to train the nn. Default dataset consist in a Nietzsch set.', required=False)
        parser.add_argument('-debug', '--debug', help='Debug mode', required=False)
        parser.add_argument('-verbose', '--verbose', help='Debug mode', required=False)

        parser.add_argument('-ss', '--sample_size', help='Number of characters to each sample', required=False)
        parser.add_argument('-ne', '--number_epochs', help='Number of epochs of training', required=False)
        parser.add_argument('-it', '--iterations', help='Number of iterarions for training', required=False)
        parser.add_argument('-fn', '--figure_name', help='Figure name for convergence graph', required=False)
        parser.add_argument('-mp', '--model_prefix', help='Model prefix for config model save', required=False)

        args = vars(parser.parse_args())

        input_data_path = str(args['dataset']) if args['dataset'] else 'data/nietzsch/input.txt'
        sample_length = int(args['sample_size']) if args['sample_size'] else 40
        n_epochs = int(args['number_epochs']) if args['number_epochs'] else 10
        iterations = int(args['iterations']) if args['iterations'] else 100

        if args['figure_name']:
            figure_name = 'src/img/%s_%s_charrnn.jpg' % (str(args['figure_name']), time.strftime('%d%m%y'))
        else:
            figure_name = 'src/img/%s_charrnn.jpg' % time.strftime('%d%m%y')

        model_prefix = str(args['model_prefix']) if args['model_prefix'] else 'lstm'
        loss_file_name = 'src/history/%s_%s.txt' % (model_prefix, time.strftime('%d%m%y'))
        debug = bool(int(args['debug'])) if args['debug'] else True
        verbose = int(args['verbose']) if args['verbose'] else 0


        # Loading dataset
        text, char_idx, idx_char = d_utils.Text().load_char_level(input_data_path)
        x_raw, y_raw = d_utils.Text().parse_dataset(text, parcel_size=sample_length, output_level=1)

        if debug:
            X = d_utils.Text().vectorization(x_raw, char_idx)[:100]
            y = d_utils.Text().vectorization(y_raw, char_idx)[:100]
        else:
            X = d_utils.Text().vectorization(x_raw, char_idx)
            y = d_utils.Text().vectorization(y_raw, char_idx)

        vocabulary_size = len(char_idx)
        loss_kf = []
        loss_train_kf = []

        print '------------------------------------------------------------------------'
        clf_rnn = charRNN(sample_length=sample_length, vocabulary_size=vocabulary_size)

        kf = KFold(len(X), 10)
        for train_index, test_index in kf:
            pass

        file_ = open(loss_file_name, 'w').close()
        print '------------------------------------------------------------------------\n\n'
        for i in range(iterations):

            print 'Iterarion [%d] Epochs [%d] | [n_epochs: %d, n_iterarions: %d] | [samples: %d]' % ((i + 1), ((i+1)*n_epochs), n_epochs, iterations, len(X))
            print 'Training...'
            train_info = clf_rnn.train(X[train_index], y[train_index], epochs=n_epochs, verbose=verbose)

            file_ = open(loss_file_name, 'a+')
            for k, loss in enumerate(train_info.history['loss']):
                __ = '%d - %f\n' % (((i * n_epochs) + k + 1), loss)
                file_.write(__)
            file_.close()

            print 'Iterarion [%d] Epochs [%d] [n_epochs: %d, n_iterarions: %d]' % ((i + 1), ((i +1) * n_epochs), n_epochs, iterations)
            print 'Evaluating...'
            loss_value = clf_rnn.evaluate(X[test_index], y[test_index])

            loss_kf.append(loss_value)
            loss_train_kf.append(np.mean(train_info.history['loss']))

            if math.isnan(np.mean(loss_kf)):
                raise 'Training will not converge, loss is NaN!'
                break

            if (i % round(iterations * 0.1)) == 0:
                model_name = 'models/%s_%s_it_%d_epochs_%d_loss_%f.h5' % (model_prefix, time.strftime('%d%m%y'),
                                                                            i, i*n_epochs, loss_value)
                clf_rnn.save(model_name)

            print '\n\nIterarion [%d] | Epoch [%d]:' % ((i + 1), ( (i+1) * n_epochs))
            print 'Training: %f | Mean: %f' % ( loss_train_kf[-1], np.mean(loss_train_kf))
            print 'Evaluation: %f | Mean: %f' % ( loss_value, np.mean(loss_kf))
            print '------------------------------------------------------------------------\n\n'

            p1 = (np.arange(np.size(loss_kf))) + 1

            # Matplotlib settings
            plt.clf()
            plt.grid()
            # Loss curve

            #   TODO:
            #       # plot per epoch
            plt.plot(p1, loss_kf, label='Loss curve evaluation (mean)', linewidth=3, color='red')
            plt.plot(p1, loss_train_kf, label='Loss curve training (mean)', linewidth=3, linestyle='--', color='green')

            plt.axis([0, (iterations + (iterations*0.1)), 0, np.max(loss_kf) + 2])
            plt.title('Convergence')
            plt.ylabel('Loss value')
            plt.xlabel('Number of iterarions with %d training epochs each' % n_epochs)

            plt.plot([0, 0], [0, np.max(loss_kf) + 1], linewidth=1, color='black')
            plt.plot([-1, (iterations + iterations * 0.1)], [0, 0], linewidth=1, color='black')
            plt.legend()
            plt.draw()
            plt.show(block=False)
            plt.savefig(figure_name)

        plt.show(block=True)
        sys.exit()

    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print ("Error :", err, exc_type, fname, exc_tb.tb_lineno)

if __name__ == '__main__':

    main()


