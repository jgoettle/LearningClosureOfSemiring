
import sys
import os

import pandas as pd

# appending neurips_experiments to PATH so we can directly execute plot_experiment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from neurips_experiments import utils
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from matplotlib import font_manager
font_dirs = ['neurips_experiments/plots/fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

label_semirings = {
        'plus-times': 'plus-times',
        'min-plus' : 'min-plus',
        'max-times' : 'max-times',
        'max-min' : 'max-min'
    }

label_cor_red = {
        '0.0' : 'without correction',
        '0.1' : 'plus 0.1',
        '0.2' : 'plus 0.2',
        '0.1_cor_minus' : 'minus 0.1',
        '0.2_cor_minus' : 'minus 0.2',
        '0.8_cor_sr-t' : 'times 0.8',
        '0.9_cor_sr-t' : 'times 0.9',
        '1.2_cor_sr-t' : 'times 1.2',
        '1.4_cor_sr-t' : 'times 1.4'
    }

label_reconstr = {
        'orig' : 'approx. closure',
        '0.1' : 'plus 0.1',
        '0.2' : 'plus 0.2',
        '1.2_cor_sr-t' : 'times 1.2',
        '1.4_cor_sr-t' : 'times 1.4'
    }

def histogram(args):
    filename, _ = utils.get_filename(parser, args)
    num_bins = args.hist_bins
    bins = np.linspace((-1) * int(args.hist_bound[0]), int(args.hist_bound[1]), int(num_bins+1))

    bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Calculate bin centers
    bin_width =bins[1] - bins[0]

    for semiring in semirings:
        hist = pd.read_csv('results/hist_{}_{}.csv'.format(semiring, filename), header=None).to_numpy().flatten()
        

        with plt.style.context('ggplot'):
            plt.rcParams['font.family'] = 'gillsans'
            plt.rcParams['xtick.color'] = 'black'
            plt.rcParams['ytick.color'] = 'black'
            plt.rcParams['axes.unicode_minus'] = False

            plt.figure()

            plt.bar(bin_centers, hist, width=bin_width, align='center',color='blue')
            if args.eval_mode == 'closure':
                plt.xlabel('$\\bar{A}$', fontsize=28, color='black')
            elif args.eval_mode == 'reduction':
                plt.xlabel('$A_{tr}}$', fontsize=28, color='black')
            elif args.eval_mode == 'plus-times-original':
                plt.xlabel('$B$', fontsize=28, color='black')
            plt.ylabel('Count of values', fontsize=28, color='black')
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.grid(axis='y', color='white')
            plt.grid(axis='x', color='#e5e5e5')
            # plt.legend(frameon=False, fontsize=18) 
            plt.tight_layout()
            plt.savefig('neurips_experiments/plots/plot_{}_{}_{}.pdf'.format(semiring, filename, variables))


def visualize(ground_truth, approximated, semiring='plus-times', filename=''):
    d = ground_truth.shape[0]
    
    with plt.style.context('ggplot'):
        plt.rcParams['font.family'] = 'gillsans'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        
        gray = cm.get_cmap('gray', 4)
        newcolors = gray(np.linspace(0, 1, 4))
        white = np.array([1, 1, 1, 1])
        black = np.array([0, 0, 0, 1])
        red = np.array([1, 0, 0, 1])
        grey = np.array([0.5, 0.5, 0.5, 1])
        newcolors[0, :] = white
        newcolors[1, :] = grey
        newcolors[2, :] = red
        newcolors[3, :] = black
        custom_cmp = ListedColormap(newcolors)

        approximated = np.where(approximated != 0, 1, 0)

        common_approximated = ground_truth * approximated
        wrong_approximated = approximated - common_approximated
        missed_approximated = ground_truth - common_approximated 
        approximated = common_approximated + 0.66 * wrong_approximated + 0.33 * missed_approximated

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(14, 4)

        ax1.imshow(ground_truth, cmap=custom_cmp)
        ax1.grid(False)
        ax1.add_patch(Rectangle((-0.5,-0.5), d - 0.15, d - 0.15, linewidth=1, edgecolor='black', facecolor='none'))
        ax1.axis('off')
        ax1.set_title('Ground Truth')

        ax2.imshow(approximated, cmap=custom_cmp)
        ax2.grid(False)
        ax2.add_patch(Rectangle((-0.5,-0.5), d - 0.15, d - 0.15, linewidth=1, edgecolor='black', facecolor='none'))
        ax2.axis('off')
        ax2.set_title(semiring)

        plt.savefig('neurips_experiments/plots/matrix_comparison/matrix_comparison_{}_{}.png'.format(filename, semiring), dpi=1000)
        plt.close(fig)


def plot_accuracy(avg, std, x_axis, elements, param='nodes', filename='default', legend=False):

    linewidth = {}
    for el in elements:
        linewidth[el] = 1.5

    with plt.style.context('ggplot'):
        plt.rcParams['font.family'] = 'gillsans'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'

        label_dic = label_cor_red if param == 'reduction' else (label_reconstr  if param == 'closure-of-reduction' else label_semirings)
        color_dic = color_cor_red if param == 'reduction' else (color_reconstr  if param == 'closure-of-reduction' else color_semirings)
        for i, label in enumerate(['SHD', 'TPR', 'SID', 'FPR', 'NNZ', 'NMSE', 'ERM']):
            if (not legend):
                upper_ob = 0
                plt.figure()
                for el in elements:
                    if(len(avg[el]) > 0):
                        plt.plot(x_axis, avg[el][:, i], label = label_dic[el], color=color_dic[el], linewidth=linewidth[el])
                        plt.fill_between(x_axis, avg[el][:, i] - std[el][:, i], avg[el][:, i] + std[el][:, i], color=color_dic[el], alpha=.1)
                        upper_ob = max(upper_ob, (avg[el][:, i] + std[el][:, i]).max())

                plt.ylabel(label, fontsize=30, color='black')
                plt.xlabel('Number of nodes', fontsize=22, color='black')
                plt.xticks(x_axis, fontsize=22)
                plt.yticks(fontsize=22)
                plt.grid(axis='y', color='white')
                plt.grid(axis='x', color='#e5e5e5')
                if label in lims.keys():
                    lower, upper = lims[label]
                    new_upper = min(upper_ob, upper) if upper_ob > lower else upper
                    plt.ylim(lower, new_upper)
                plt.tight_layout()
                print('neurips_experiments/plots/plot_{}_{}.pdf'.format(filename, file_label[label]))
                plt.savefig('neurips_experiments/plots/plot_{}_{}.pdf'.format(filename, file_label[label]))

            # only print legend
            elif (i == 0):
                plt.figure()

                plt.rcParams['axes.facecolor']='white'
                plt.rcParams['savefig.facecolor']='white'
                for el in elements:
                    if(len(avg[el] > 0)):
                        plt.plot([], [], label = label_dic[el], color=color_dic[el], linewidth=linewidth[el])
                plt.xticks([])
                plt.yticks([])
                plt.legend(frameon=False, fontsize=20)
                plt.tight_layout()
                plt.savefig('neurips_experiments/plots/plot_{}_legend_only.pdf'.format( filename), bbox_inches='tight')
        


def plot_accuracy_vs_param(args, param='sparsity'):
    dic = vars(args)
    
    avg = {}
    std = {}


    if param == 'nodes' or 'reduction' or 'closure-of-reduction':
        values = dic['nodes']
    elif param == 'samples':
        values = dic['samples']
    else:
        print("case not covered")

    if param == 'nodes' or param == 'samples':

        for key in semirings:
            avg[key] = []
            std[key] = []
        # finding the name of the output files according to the experimental settings 
        filename, _ = utils.get_filename(parser, args)

        with open('results/{}.csv'.format(filename), 'r') as f:
            for line in f:
                info = line.split(',')
                for semiring in semirings:
                    if(info[0] == 'Acc {} is'.format(semiring)):
                        avg[semiring].append([float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[6]), float(info[7]), float(info[8])])
                    elif(info[0] == 'Std {} is'.format(semiring)):
                        std[semiring].append([float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[6]), float(info[7]), float(info[8])])

        for semiring in semirings:
            avg[semiring] = np.array(avg[semiring])
            std[semiring] = np.array(std[semiring])
        elements = semirings

    elif param=='reduction' or param=='closure-of-reduction':

        
        # finding the name of the output files according to the experimental settings 
        filename, _ = utils.get_filename(parser, args)
        elements = label_cor_red.keys() if param=='reduction' else label_reconstr.keys()

        for cor in elements:
            
            temp_filename = '{}cor_val_{}_'.format(filename, cor) if cor!='orig'  else '{}cor_{}_'.format(filename, cor)
    
            avg[cor] = []
            std[cor] = []
            with open('results/{}.csv'.format(temp_filename), 'r') as f:
                for line in f:
                    info = line.split(',')
                    semiring = semirings[0]
                    if(info[0] == 'Acc {} is'.format(semiring)):
                        avg[cor].append([float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[6]), float(info[7]), float(info[8])])
                    elif(info[0] == 'Std {} is'.format(semiring)):
                        std[cor].append([float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[6]), float(info[7]), float(info[8])])


            avg[cor] = np.array(avg[cor])
            std[cor] = np.array(std[cor])

    x_axis = values
    plot_accuracy(avg, std, x_axis, elements, param=param, filename=filename, legend=(args.legend == 'True'))


if __name__ == '__main__':
    parser, args = utils.get_args()

    samples = args.samples
    variables =  args.nodes
    noise = args.noise
    runs = args.runs 
    (a, b) = tuple(args.weight_bounds)
    k = args.edges
    semirings = args.semirings

    color_semirings = {
        'plus-times': 'black', 
        'min-plus' : 'cyan',
        'max-times' : 'C7',
        'max-min' : 'C2'
    }

    color_cor_red = {
        '0.0' : 'black',
        '0.1' : 'C5', 
        '0.2' : 'navy',
        '0.1_cor_minus' : 'green',
        '0.2_cor_minus' : 'C4', 
        '0.8_cor_sr-t' :  'sienna', 
        '0.9_cor_sr-t' : 'purple', 
        '1.2_cor_sr-t' : 'teal',
        '1.4_cor_sr-t' : 'pink',
    }

    color_reconstr= {
        'orig' : 'black',
        '0.1' : 'C5', 
        '0.2' : 'navy', 
        '1.2_cor_sr-t' : 'teal',
        '1.4_cor_sr-t' : 'pink',
    }

    file_label = {
        'SHD' : 'shd',
        'TPR' : 'tpr',
        'NNZ' : 'nnz', 
        'SID' : 'sid',
        'FPR' : 'fpr',
        'NMSE': 'nmse',
        'ERM': 'erm'
    }               

    lims = { 
        'SHD' : [0, 500],
        'ERM' : [0, 50]
    }
    if args.hist == "True":
        histogram(args)  
    
    if args.eval_mode == 'reduction':
        plot_accuracy_vs_param(args, param='reduction')
    elif args.eval_mode == 'closure-of-reduction':
        plot_accuracy_vs_param(args, param='closure-of-reduction') 
    elif len(args.nodes) > 1:
        plot_accuracy_vs_param(args, param='nodes')

