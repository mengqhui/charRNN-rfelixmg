from math import *
from matplotlib import pyplot as plt
print plt.get_backend()



def main():
    x = range(-50, 51, 1)

    for pow in range(1,5):   # plot x^1, x^2, ..., x^4

        y = [Xi**pow for Xi in x]
        print y

        plt.plot(x, y)
        plt.draw()
        #plt.show()             #this plots correctly, but blocks execution.
        plt.show(block=False)   #this creates an empty frozen window.
        _ = raw_input("Press [enter] to continue.")

        plt.pause(1)


if __name__ == '__main__':
    main()



    # for iteration in range(10):
    #
    #
    #
    #
    #
    #     # TODO:
    #     #  * measure time
    #     t_st = time.time()
    #     clf_rnn.train(X, y, epochs=10)
    #     print 'Time training %f seconds' % (time.time() - t_st)
    #
    #     start_index = random.randint(0, len(text) - sample_length - 1)
    #
    #     for diversity in [0.2]: #[0.2, 0.5, 1.0, 1.2]:
    #         print '----------------------------------------------------------------------'
    #         print ">> Iteration [%d] | Diversity [%f]" % (iteration, diversity)
    #
    #         generated = ''
    #         sentence = text[start_index: start_index + sample_length]
    #         generated += sentence
    #
    #         print 'Seed:       "%s"' % sentence
    #         print 'Expected:   "%s"' % text[start_index: start_index + sample_length + 1]
    #         #sys.stdout.write(generated)
    #
    #         next_sentence = ''
    #         for i in range(400):
    #             x = np.zeros((1, sample_length, vocabulary_size))
    #             for t, char in enumerate(sentence):
    #                 x[0, t, char_idx[char]] = 1.
    #
    #             preds = clf_rnn.predict(x)
    #             if i % 150 == 0:
    #                 print 'Prediction: ', idx_char[np.argmax(preds)]
    #             next_index = clf_rnn.sample(preds, diversity)
    #             next_char = idx_char[next_index]
    #
    #             generated += next_char
    #             sentence = sentence[1:] + next_char
    #
    #             next_sentence += next_char
    #             #print '>> Next_char [%d]: "%s"' %( i+1, sentence)
    #
    #             #sys.stdout.write(next_char)
    #             #sys.stdout.flush()
    #         #print 'Prediction: <<<"%s">>>' % next_sentence
    #