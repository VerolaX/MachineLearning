#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('_em_aspect.py'):
    print('ERROR: This file is not named correctly! Please name it as LastName_em_aspect.py (replacing LastName with your last name)!')
    exit(1)

# DATA_PATH = "/u/cs246/data/em/" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)
DATA_PATH ='/Users/Robert/Desktop/MachineLearning/HW7/'

def parse_data(args):
    num = int
    dtype = np.int
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
        lambdas = np.zeros(args.cluster_num)
        alphas = np.zeros((args.cluster_num,10))
        betas = np.zeros((args.cluster_num,10))
        #TODO: randomly initialize clusters (lambdas, alphas, and betas)
        raise NotImplementedError #remove when random initialization is implemented
    else:
        lambdas = []
        alphas = []
        betas = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #lambda a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 b0 b1 b2 b3 b4 b5 b6 b7 b8 b9
                tokens = list(map(float,line.split()))
                lambda_k = tokens[0]
                alpha_k = tokens[1:11]
                beta_k = tokens[11:]
                lambdas.append(lambda_k)
                alphas.append(alpha_k)
                betas.append(beta_k)
        lambdas = np.asarray(lambdas)
        alphas = np.asarray(alphas)
        betas = np.asarray(betas)
        args.cluster_num = len(lambdas)

    #TODO: do whatever you want to pack the lambdas, alphas, and betas into the model variable (just a tuple, or a class, etc.)
    model = None
    raise NotImplementedError #remove when model initialization is implemented
    return model

def train_model(model, train_xs, dev_xs, args):
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)
    raise NotImplementedError #remove when model training is implemented
    return model

def average_log_likelihood(model, data, args):
    from math import log
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    ll = 0.0
    raise NotImplementedError #remove when average log likelihood calculation is implemented
    return ll

def extract_parameters(model):
    #TODO: extract lambdas, alphas, and betas from the model and return them (same type and shape as in init_model)
    lambdas = None
    alphas = None
    betas = None
    raise NotImplementedError #remove when parameter extraction is implemented
    return lambdas, alphas, betas

def main():
    import argparse
    import os
    print('Aspect') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of pairs.')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'pairs.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    args = parser.parse_args()

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    ll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, alphas, betas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Alphas: {}'.format(intersperse(' | ')(map(intersperse(' '),alphas))))
        print('Betas: {}'.format(intersperse(' | ')(map(intersperse(' '),betas))))

if __name__ == '__main__':
    main()