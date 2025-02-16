import numpy as np
import networkx as nx
import argparse
import random

from threading import Thread
import functools

def get_args():
    """
    Adapted from SparceRC
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', default='gauss', type=str, help='Choices none, gauss, gumbel, uniform')
    parser.add_argument('--noise_std', default=0.01, type=float, help='Noise magnitude')
    parser.add_argument('--noise_effect', default='both', type=str, help='Where the noise is applied. Choices: signal/root_causes')
    parser.add_argument('--sparsity', default=0.1, type=float, help='Probability of data being nonzero at vertex v')
    parser.add_argument('--omega', default=0.09, type=float, help='Thresholding the output matrix of sparserc')

    parser.add_argument('--semirings', default=['plus-times', 'min-plus', 'max-times', 'max-min'], nargs='+', type=str, help='semiring used to compute the transitive closure')
    parser.add_argument('--eval_mode', default='closure', type=str, help='Choices closure, reduction, closure-of-reduction, plus-times-original, root-causes')

    parser.add_argument('--hist', default='False', type=str, help='Whether to generate data for the value distribution, not compatible with mutiple samples/nodes')
    parser.add_argument('--hist_bins', default=140, type=int, help='Number of bins of the histogram')
    parser.add_argument('--hist_bound', default=[7,7], nargs=2, type=int, help='What are the boundaries of the histogram, min is negated ')

    parser.add_argument('--weight_bounds', default=[0.1, 0.9], nargs='+', type=float, help='initialization of weighted adjacency matrix')
    parser.add_argument('--edges', default=4, type=int, help='graph has k * d edges')

    parser.add_argument('--samples', default=[5000], nargs='+', type=int, help='number of samples')
    parser.add_argument('--nodes', default=[100], nargs='+', type=int, help='number of graph vertices to consider')
    parser.add_argument('--graph_type', default='ER', type=str, help='Choices ER (Erdös-Renyi), SF (Scale Free)')
    parser.add_argument('--runs', default=5, type=int, help="how many times to generate the random DAG and run the methods") 

    parser.add_argument('--timeout', default=1000, type=int, help='Total allowed runtime for a method')

    #TODO
    parser.add_argument('--table', default='TPR', type=str, help='Choices TPR, SHD')
    parser.add_argument('--sparserc_epochs', default=15000, type=int, help="Number of training epochs of model")
    parser.add_argument('--legend', default='False', type=str, help='Whether to plot the legend only')

    parser.add_argument('--cor_val', default=0.05, type=float, help='transitive reduction correction value')
    parser.add_argument('--cor', default='plus', type=str, help='Reduction correction, Choices none, plus, sr-t/semiring-times, plus, orig (outputs the orig. extimated closure)')

    args = parser.parse_args()

    return parser, args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def timeout(timeout):
    '''
    Timeout function utility
    from: https://stackoverflow.com/questions/21827874/timeout-a-function-windows

    Use: MyResult = timeout(timeout=16)(MyModule.MyFunc)(MyArgs)
    '''
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print ('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco


def is_dag(W):
    G = G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    return nx.is_directed_acyclic_graph(G)

def get_filename(parser, args):
    # naming the output files according to the experimental settings 
    dic = vars(args)
    filename = ''
    label = ''
    for key in dic.keys():
        if(key not in ['variables', 'legend'] and dic[key]!= parser.get_default(key)):
            filename += '{}_{}_'.format(key, dic[key])

        label += '{} = {}, '.format(key, dic[key])
    filename = filename if len(filename) > 0 else 'default'
    return filename, label
