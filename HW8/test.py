#!/usr/bin/env python3
import numpy as np
import time
# if not __file__.endswith('_hmm_gaussian.py'):
#     print('ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
#     exit(1)

# DATA_PATH = "/u/cs246/data/em/" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)
DATA_PATH = "/Users/Robert/Desktop/MachineLearning/HW8/"
# DATA_PATH = '/home/tianyou/MachineLearning/HW7/'

iteration_list = list([1,10,20,50])
cluster_numbers = list([2,3,4,5,10,20])

class Model:
    def __init__(self, initials, transitions, mus, sigmas):
        self.initials = initials
        self.transitions = transitions
        self.mus = mus
        self.sigmas = sigmas


def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    if args.cluster_num:
        mus = np.zeros((args.cluster_num,2))
        if not args.tied:
            sigmas = np.zeros((args.cluster_num,2,2))
        else:
            sigmas = np.zeros((2,2))
        transitions = np.zeros((args.cluster_num,args.cluster_num)) #transitions[i][j] = probability of moving from cluster i to cluster j
        initials = np.zeros(args.cluster_num) #probability for starting in each state
        #TODO: randomly initialize clusters (mus, sigmas, initials, and transitions)
        mus = np.random.rand(args.cluster_num, 2)
        if not args.tied:
            for c in range(args.cluster_num):
                rand_sigma = list(np.random.rand(3))
                for i in range(2):
                    for j in range(i, 2):
                        sigmas[c][i][j] = rand_sigma.pop()
                for i in range(1, 2):
                    for j in range(i):
                        sigmas[c][i][j] = sigmas[c][j][i]
                sigmas[c] += np.identity(2)

        else:
            rand_sigmas = list(np.random.rand(1000))
            for i in range(2):
                for j in range(i, 2):
                    sigmas[i][j] = rand_sigmas.pop()
            for i in range(1, 2):
                for j in range(i):
                    sigmas[j][i] = sigmas[i][j]
            sigmas += np.identity(2)

        rand = np.random.rand(args.cluster_num)
        initials = rand/sum(rand)
        transitions = np.random.rand(args.cluster_num,args.cluster_num)
        # raise NotImplementedError #remove when random initialization is implemented
    else:
        mus = []
        sigmas = []
        transitions = []
        initials = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
                vals = list(map(float,line.split()))
                initials.append(vals[0])
                mus.append(vals[1:3])
                sigmas.append([vals[3:5],vals[5:7]])
                transitions.append(vals[7:])
        initials = np.asarray(initials)
        transitions = np.asarray(transitions)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(initials)

    #TODO: Do whatever you want to pack mus, sigmas, initals, and transitions into the model variable (just a tuple, or a class, etc.)
    model = Model(initials, transitions, mus, sigmas)
    # raise NotImplementedError #remove when model initialization is implemented
    return model

def forward(model, data, args):
    from scipy.stats import multivariate_normal
    from math import log
    alphas = np.zeros((len(data),args.cluster_num))
    log_likelihood = 0.0
    #TODO: Calculate and return forward probabilities (normalized at each timestep; see next line) and log_likelihood
    #NOTE: To avoid numerical problems, calculate the sum of alpha[t] at each step, normalize alpha[t] by that value, 
    # and increment log_likelihood by the log of the value you normalized by. This will prevent the probabilities from going to 0, 
    # and the scaling will be cancelled out in train_model when you normalize (you don't need to do anything different than what's in the notes). 
    # This was discussed in class on April 3rd.

    initials, transitions, mus, sigmas = extract_parameters(model)

    for t in range(0, len(data)):
        if t == 0:
            for c in range(args.cluster_num):
                if not args.tied:
                    alphas[0, c] = initials[c] * multivariate_normal(mean=mus[c], cov=sigmas[c]).pdf(data[0])
                else:
                    alphas[0, c] = initials[c] * multivariate_normal(mean=mus[c], cov=sigmas).pdf(data[0])
        else:            
            for i in range(args.cluster_num):
                for j in range(args.cluster_num):
                    if not args.tied:
                        alphas[t, i] += alphas[t-1, j] * transitions[j, i] * multivariate_normal(mean=mus[i], cov=sigmas[i]).pdf(data[t])
                    else:
                        alphas[t, i] += alphas[t-1, j] * transitions[j, i] * multivariate_normal(mean=mus[i], cov=sigmas).pdf(data[t])
        
        log_likelihood += log(np.sum(alphas[t, :]))
        alphas[t, :] /= np.sum(alphas[t, :])


    # raise NotImplementedError
    return alphas, log_likelihood

def backward(model, data, args):
    from scipy.stats import multivariate_normal
    betas = np.zeros((len(data), args.cluster_num))
    #TODO: Calculate and return backward probabilities (normalized like in forward before)
    
    betas[len(data)-1, :] = 1
    betas[len(data)-1, :] /= np.sum(betas[len(data)-1, :])

    initials, transitions, mus, sigmas = extract_parameters(model)

    for t in range(len(data)-2, -1, -1):
        for i in range(args.cluster_num):
            for j in range(args.cluster_num):
                if not args.tied:
                    betas[t, i] += transitions[i, j] * betas[t+1, j] * multivariate_normal(mean=mus[j], cov=sigmas[j]).pdf(data[t+1])
                else:
                    betas[t, i] += transitions[i, j] * betas[t+1, j] * multivariate_normal(mean=mus[j], cov=sigmas).pdf(data[t+1])
        betas[t, :] /= np.sum(betas[t, :])
    # raise NotImplementedError
    return betas

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)
    initials, transitions, mus, sigmas = extract_parameters(model)
    num_data = train_xs.shape[0]

    for iter in range(args.iterations):
        # E step:
        gamma = np.zeros((num_data, args.cluster_num))
        xi = np.zeros((num_data, args.cluster_num, args.cluster_num))

        alphas, _ = forward(model, train_xs, args)
        betas = backward(model, train_xs, args)

        for t in range(num_data):
            for c in range(args.cluster_num):
                gamma[:, c] = np.multiply(alphas[:, c], betas[:, c]) / np.sum(np.multiply(alphas, betas), axis=1)

            for i in range(args.cluster_num):
                for j in range(args.cluster_num):
                    if t != 0:
                        if not args.tied:
                            xi[t, i, j] = alphas[t-1, i] * transitions[i, j] * multivariate_normal(mean=mus[j], cov=sigmas[j]).pdf(train_xs[t]) \
                            * betas[t, j]
                        else:
                            xi[t, i, j] = alphas[t-1, i] * transitions[i, j] * multivariate_normal(mean=mus[j], cov=sigmas).pdf(train_xs[t]) \
                            * betas[t, j]
            if t != 0:
                xi[t,:,:] /= np.sum(xi[t,:,:])

        # M step
        initials = gamma[0, :]
        for i in range(args.cluster_num):
            mus[i] = np.dot(gamma[:, i], train_xs) / np.sum(gamma[:, i])

            if not args.tied:
                sigmas[i] = np.dot(gamma[:, i] * (train_xs - mus[i]).T, (train_xs - mus[i])) / np.sum(gamma[:, i])
            else:
                sigmas += np.dot(gamma[:, i] * (train_xs - mus[i]).T, (train_xs - mus[i])) / np.sum(gamma[:, i]) / args.cluster_num

            for j in range(args.cluster_num):
                transitions[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:, i])

        model = Model(initials, transitions, mus, sigmas)

    if args.tied:
        model.sigmas = model.sigmas / args.iterations
    # raise NotImplementedError #remove when model training is implemented
    return model

def average_log_likelihood(model, data, args):
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    #NOTE: yes, this is very simple, because you did most of the work in the forward function above
    ll = 0.0
    _, ll = forward(model, data, args)
    ll = ll / len(data)
    # raise NotImplementedError #remove when average log likelihood calculation is implemented
    return ll

def extract_parameters(model):
    #TODO: Extract initials, transitions, mus, and sigmas from the model and return them (same type and shape as in init_model)
    initials = model.initials
    transitions = model.transitions
    mus = model.mus
    sigmas = model.sigmas
    # raise NotImplementedError #remove when parameter extraction is implemented
    return initials, transitions, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
    # init_group = parser.add_mutually_exclusive_group(required=True)
    # init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    # init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    parser.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    ll_trains = []
    ll_devs = []

    for cluster_num in cluster_numbers:
        ll_train = []
        ll_dev = []
        for iters in iteration_list:
            if args.tied:
                print("tied")
            start = time.time()
            args.iterations = iters
            args.cluster_num = cluster_num

            print("Cluster Number: ", args.cluster_num, "|| Iterations: ", args.iterations)
            train_xs, dev_xs = parse_data(args)
            model1 = init_model(args)
            model2 = init_model(args)
            model3 = init_model(args)
            # model4 = init_model(args)
            # model5 = init_model(args)

            model1 = train_model(model1, train_xs, dev_xs, args)
            model2 = train_model(model2, train_xs, dev_xs, args)
            model3 = train_model(model3, train_xs, dev_xs, args)
            # model4 = train_model(model4, train_xs, dev_xs, args)
            # model5 = train_model(model5, train_xs, dev_xs, args)

            ll_train1 = average_log_likelihood(model1, train_xs, args)
            ll_train2 = average_log_likelihood(model2, train_xs, args)
            ll_train3 = average_log_likelihood(model3, train_xs, args)
            # ll_train4 = average_log_likelihood(model4, train_xs, args)
            # ll_train5 = average_log_likelihood(model5, train_xs, args)

            average_ll_train = (ll_train1+ll_train2+ll_train3) / 3
            ll_train.append(average_ll_train)
            print('Train LL: {}'.format(average_ll_train))
            if not args.nodev:
                ll_dev1 = average_log_likelihood(model1, dev_xs, args)
                ll_dev2 = average_log_likelihood(model2, dev_xs, args)
                ll_dev3 = average_log_likelihood(model3, dev_xs, args)
                average_ll_dev = (ll_dev1+ll_dev2+ll_dev3) / 3
                ll_dev.append(average_ll_dev)
                print('Dev LL: {}'.format(average_ll_dev))

            print("Time: ", time.time() - start, "\n")
            ll_trains.append(ll_train)
            ll_devs.append(ll_dev)

    import seaborn as sns
    import matplotlib.pyplot as plt

    f, axarr = plt.subplots(2, sharex=True)
    # f2, axarr2 = plt.subplots(2, sharex=True)

    sns.set()
    pal = sns.color_palette("Set3", 9)
    # pal2 = sns.color_palette("Set3", 9)

    for i in range(len(cluster_numbers)):
        for ll_train in ll_trains:
            axarr[0].plot(iterations, ll_train, c=pal[i], label='cluster_num={}'.format(cluster_num[i]))
            axarr[0].set_ylim(-4.9, -3.5)
            axarr[0].legend(loc='upper center', prop={'size': 6}, bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
            axarr[0].set_title('Training')
        for ll_dev in ll_devs:
            axarr[1].plot(iterations, ll_dev, c=pal[i], label='cluster_num={}'.format(cluster_nums[i]))
            axarr[1].set_title('Dev')

    if args.tied:
        f.savefig('tied3.png', dpi=300)
    else:
        f.savefig('not_tied3.png', dpi=300)

    initials, transitions, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
        print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '),transitions))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()