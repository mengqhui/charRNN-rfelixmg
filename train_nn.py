# http://stackoverflow.com/questions/7427101/dead-simple-argparse-example-wanted-1-argument-3-results
# https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py#L74

import os
from samba.netcmd import time
from tornado.platform import epoll
import math

from keras.models import model_from_config
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

        # Getting arg parameters
        parser = argparse.ArgumentParser(description='Training LSTM to generate text in char-rnn level')
        parser.add_argument('-d', '--dataset', help='Input new dataset to train the nn. Default dataset consist in a Nietzsch set.', required=False)
        parser.add_argument('-debug', '--debug', help='Debug mode', required=False)
        parser.add_argument('-verbose', '--verbose', help='Debug mode', required=False)

        parser.add_argument('-ss', '--sample_size', help='Number of characters to each sample', required=False)
        parser.add_argument('-ne', '--number_epochs', help='Number of epochs of training', required=False)
        parser.add_argument('-it', '--iterations', help='Number of iterarions for training', required=False)
        parser.add_argument('-fn', '--figure_name', help='Figure name for convergence graph', required=False)
        parser.add_argument('-mp', '--model_prefix', help='Model prefix for config model save', required=False)
        parser.add_argument('-seed', '--seed_dataset', help='Load seed to split dataset', required=False)
        parser.add_argument('-lr', '--learning_rate', help='Learning rate, default=0.001', required=False)

        args = vars(parser.parse_args())

        input_data_path = str(args['dataset']) if args['dataset'] else 'data/tinyshakespeare/input.txt'

        # Length for each sample (number of characters input to the network)
        sample_length = int(args['sample_size']) if args['sample_size'] else 26

        # Number of iterarions
        iterations = int(args['iterations']) if args['iterations'] else 100

        learning_rate = float(args['learning_rate']) if args['learning_rate'] else 0.01

        # Number of epochs per iteration
        n_epochs = int(args['number_epochs']) if args['number_epochs'] else 10
        total_epochs = iterations * n_epochs


        model_prefix = str(args['model_prefix']) if args['model_prefix'] else 'lstm'

        # File name instances to save reports
        #   # Writing report loss file
        loss_file_name = 'src/history/%s_%s.txt' % (model_prefix, time.strftime('%d%m%y'))
        open(loss_file_name, 'w').close()

        if args['figure_name']:
            figure_name = 'src/img/%s_%s_%s.jpg' % (str(args['figure_name']), time.strftime('%d%m%y'), model_prefix)
        else:
            figure_name = 'src/img/%s_charrnn.jpg' % (time.strftime('%d%m%y'), model_prefix)

        debug = bool(int(args['debug'])) if args['debug'] else True
        verbose = int(args['verbose']) if args['verbose'] else 0

        seed_db = str(args['seed_dataset']) if args['seed_dataset'] else False


        # Loading and preparing dataset

        text, char_idx, idx_char = d_utils.Text().load_char_level(input_data_path)
        x_raw, y_raw = d_utils.Text().parse_dataset(text, parcel_size=sample_length, step=10, output_level=1)

        if debug:
            X = d_utils.Text().vectorization(x_raw, char_idx)[:100]
            y = d_utils.Text().vectorization(y_raw, char_idx)[:100]
        else:
            X = d_utils.Text().vectorization(x_raw, char_idx)
            y = d_utils.Text().vectorization(y_raw, char_idx)

        # Number of chars in the dataset
        vocabulary_size = len(char_idx)


        # Statistical information to be plotted in the graphich
        loss_kf = [] #List of loss values for evaluation
        loss_train_kf = [] #List of loss values for training
        plot_loss_evaluation = [] #List of loss values for evaluation plotting
        plot_loss_train = [] #List of loss values for training plotting
        plot_lr_rate = [[0, 0, learning_rate]] #List of loss values for evaluation
        fig = plt.figure(figsize=(48, 32))

        print '------------------------------------------------------------------------'

        # Creating model charRNN
        clf_rnn = charRNN(sample_length=sample_length, vocabulary_size=vocabulary_size, hidden_states=128, learning_rate=learning_rate)

        # Loading seed to split dataset
        if seed_db:
            train_index, test_index = d_utils.Text.load_split(seed_db)
        else:
            # Split dataset in 50% training 50% testing
            train_index, test_index = d_utils.Text().split_dataset(len(X), 0.5)

            # saving seed of dataset split
            split_file_name = 'src/%s_%s' % (model_prefix, time.strftime('%d%m%y'))
            d_utils.Text().save_split(split_file_name, (train_index, test_index))

        print '------------------------------------------------------------------------\n\n'
        # Running iterations
        for i in range(iterations):

            print 'Iterarion [%d] Epochs [%d] | [n_epochs: %d, n_iterarions: %d] | [samples: %d]' % \
                  ((i + 1), ((i+1)*n_epochs), n_epochs, iterations, len(X))
            print 'Training...'
            # Training model
            train_info = clf_rnn.train(X[train_index], y[train_index], epochs=n_epochs, verbose=verbose)

            print 'Iterarion [%d] Epochs [%d] [n_epochs: %d, n_iterarions: %d]' % \
                  ((i + 1), ((i +1) * n_epochs), n_epochs, iterations)
            print 'Evaluating...'
            # Evaluating model
            loss_value = clf_rnn.evaluate(X[test_index], y[test_index])



            # Saving breef report regarding loss value per epoch
            # It's necessary to have loss_value & training_loss value to run this part
            file_ = open(loss_file_name, 'a+')
            for k, loss in enumerate(train_info.history['loss']):
                __ = '%d - %f\n' % (((i * n_epochs) + k + 1), loss)
                plot_loss_train.append(loss)
                plot_loss_evaluation.append(loss_value)
                file_.write(__)
            file_.close()

            # Appending loss value for evaluation
            loss_kf.append(loss_value)
            # Appending loss value for Training
            loss_train_kf.append(np.mean(train_info.history['loss']))

            # In case of Gradient Explode
            if math.isnan(np.mean(loss_kf)):
                raise 'Training will not converge, loss is NaN!'
                break

            # Every 10% of the processing:
            #   # saving model
            #   # changing learning rate
            if (i * n_epochs) % (total_epochs * 0.1) == 0:
                # Appending new learning rate
                if i > 0:
                    # Decrease learning rate
                    lr = clf_rnn.update_parameter(lr_decrease=0.1)
                    plot_lr_rate.append([(i+1)*n_epochs, loss_train_kf[-1], lr])

                    print 'Decrease learning rate to %f' % lr

                # Saving models
                model_name = 'models/%s_%s_it_%d_epochs_%d_loss_%f.h5' % (model_prefix, time.strftime('%d%m%y'),
                                                                            i, i*n_epochs, loss_value)
                clf_rnn.save(model_name)

                print 'Saving model progress...'

            print '\n\nIterarion[%d]|Epoch[%d]:' % ((i+1), (i+1)*n_epochs)

            print 'Training: %f | Mean: %f' % (loss_train_kf[-1], np.mean(loss_train_kf))
            print 'Evaluation: %f | Mean: %f' % (loss_value, np.mean(loss_kf))
            print '------------------------------------------------------------------------\n\n'


            # Matplotlib settings
            plt.clf()
            plt.grid()
            plt.axis([0, (total_epochs + (total_epochs * 0.1)), 0, np.max(plot_loss_train) + 2])
            plt.title('Convergence')
            plt.ylabel('Loss value')
            plt.xlabel('Epochs [%s]' % total_epochs)
            plt.plot([0, 0], [0, np.max(loss_kf) + 1], linewidth=1, color='black')
            plt.plot([-1, (iterations + iterations * 0.1)], [0, 0], linewidth=1, color='black')

            # Plotting loss training x evaluation
            p1 = (np.arange(np.size(plot_loss_evaluation))) + 1
            plt.plot(p1, plot_loss_evaluation, label='Loss evaluation (mean)', linewidth=3, color='red')
            plt.plot(p1, plot_loss_train, label='Loss training (mean per epoch)', linewidth=3, linestyle='--', color='green')

            # Plot learning rates
            if len(np.shape(plot_lr_rate)) > 1:
                for (x_tmp, y_tmp, lr_value) in plot_lr_rate:
                    plt.plot(x_tmp, y_tmp, 'ro', color='blue')
                    lr_label = 'lr: %.1g' % lr_value
                    plt.annotate(lr_label,
                                 xy=(x_tmp, y_tmp),
                                 xytext=(x_tmp+1, y_tmp+1),
                                 bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.5),
                                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            plt.legend()
            plt.draw()
            plt.show(block=False)
            plt.savefig(figure_name)

        # Releasing figure
        plt.show(block=True)
        sys.exit()

    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print ("Error :", err, exc_type, fname, exc_tb.tb_lineno)

if __name__ == '__main__':
    random.seed(0)
    main()


