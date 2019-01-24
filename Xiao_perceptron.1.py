#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#TODO: understand that you should not need any other imports other than those already in this file; if you import something that is not installed by default on the csug machines, your code will crash and you will lose points

NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "/u/cs246/data/adult/" #TODO: if you are working somewhere other than the csug server, change this to the directory where a7a.train, a7a.dev, and a7a.test are on your machine
DATA_PATH = '/Users/Robert/Desktop/adult'

#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value
    x[-1] = 1 #bias
    return y, x

#return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray(ys), np.asarray(xs) #returns a tuple, first is an array of labels, second is an array of feature vectors

def perceptron(train_ys, train_xs, dev_ys, dev_xs, args):
    weights = np.zeros(NUM_FEATURES)
    best = np.zeros(NUM_FEATURES)
    best_index = 0
    max_acc = 0
    f, axarr = plt.subplots(2, sharex=True)
    lr_arr = [0.001, 0.01, 0.05, 0.1, 1]
    for l in range(len(lr_arr)):
        acc_dev = list()
        acc_train = list()
    #TODO: implement perceptron algorithm here, respecting args
        for k in range(args.iterations):
            for n in range(train_ys.size):
                if (train_ys[n] * np.dot(train_xs[n,:].reshape(1,-1), weights.reshape(-1,1))) <= 0:
                    weights = weights + lr_arr[l] * train_ys[n] * train_xs[n,:]
                    if k == 0:
                        best = weights
                        max_acc = test_accuracy(weights, train_ys, train_xs)
            acc_train.append(test_accuracy(weights, train_ys, train_xs))
            if not args.nodev:
                acc_dev.append(test_accuracy(weights, dev_ys, dev_xs))
                if (k > 0) and (acc_dev[k] > max_acc):
                    best = weights
                    best_index = k
                    max_acc = acc_dev[k]
        
        if not args.nodev:
            '''
            x = range(1, args.iterations+1)
            plt.plot(x, acc_train, 'r--', label = 'train')
            plt.plot(x, acc_dev, 'g--', label = 'dev')
            plt.ylim(0,1)
            plt.legend(loc = 'lower right')
            plt.show()
            print('Best number of iterations at learning rate = %s is %s' % (args.lr, best_index+1))
            '''
            x = range(1, args.iterations+1)
            sns.set()
            pal = sns.color_palette("Set2", 5)
            axarr[0].plot(x, acc_train, c=pal[l], label='learning rate = {}'.format(lr_arr[l]), linewidth=1)
            axarr[0].legend(loc='lower right')
            axarr[0].set_ylim(0.2,1)
            axarr[0].set_title('training')

            axarr[1].plot(x, acc_dev, c=pal[l], label='learning rate = {}'.format(lr_arr[l]), linewidth=1)
            axarr[1].legend(loc='lower right')
            axarr[1].set_ylim(0.2,1)
            axarr[1].set_title('development')

    plt.show()

    if not args.nodev:
        return best
        
    return weights

def test_accuracy(weights, test_ys, test_xs):
    accuracy = 0.0
    #TODO: implement accuracy computation of given weight vector on the test data (i.e. how many test data points are classified correctly by the weight vector)
    result = np.dot(test_xs, weights.T)
    result = np.sign(result)
    pos = 0
    for i in range(result.size):
        if result[i] == test_ys[i]:
            pos = pos + 1
    #for i in range(test_ys.size):
    accuracy = pos / test_ys.size
    return accuracy

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate to use for update in training loop.')
    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')
    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.train_file: str; file name for training data.
    args.dev_file: str; file name for development data.
    args.test_file: str; file name for test data.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)
    weights = perceptron(train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(weights, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    print('Feature weights (bias last): {}'.format(' '.join(map(str,weights))))

if __name__ == '__main__':
    main()
