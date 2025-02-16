import sys
import os

# appending neurips_experiments to PATH so we can directly execute plot_experiment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from neurips_experiments import utils

if __name__ == '__main__':
    parser, args = utils.get_args()
    semirings = args.semirings

    # finding the name of the output files according to the experimental settings 
    dic = vars(args)
    filename = ''

    for key in dic.keys():
        if(key not in ['table'] and dic[key]!= parser.get_default(key)):
            filename += '{}_{}_'.format(key, dic[key])

    if len(filename) == 0:
        filename = 'default'


    avg = {}
    std = {}
    for key in semirings:
        avg[key] = []
        std[key] = []
    
    with open('results/{}.csv'.format(filename), 'r') as f:
        for line in f:
            info = line.split(',')
            for semiring in semirings:
                if(info[0] == 'Acc {} is'.format(semiring)):
                    avg[semiring].append([float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[6]), float(info[7]), float(info[8])])
                elif(info[0] == 'Std {} is'.format(semiring)):
                    std[semiring].append([float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[6]), float(info[7]), float(info[8])])
    
                

    latest = len(avg[semirings[0]]) - 1
    best_tpr = max([avg[s][latest][2] for s in semirings])
    best_shd = min([avg[s][latest][0] for s in semirings])
    best_sid = min([avg[s][latest][1] for s in semirings])
    best_time = min([avg[s][latest][6] for s in semirings])

    result = ''
    d = args.nodes[-1]
    for semiring in semirings:
        result += '{} '.format(semiring)
        
        SHD_fail = args.edges * d
        SID_fail = args.edges * d
        TPR_fail = 0.8
        FPR_fail = 0.2
        NNZ_fail = d * d # cannot fail
        NMSE_fail = 0.3  
        EDGES_rem_fail = d

        # SHD, TPR, SID, FPR, NNZ, NMSE, Edges RM
        for ind,fail_cutoff in enumerate([SHD_fail, TPR_fail,  SID_fail, FPR_fail, NNZ_fail, NMSE_fail, EDGES_rem_fail]):
            h_avg = avg[semiring][latest][ind]
            h_std = std[semiring][latest][ind]
            if ind != 4:
                if(ind != 1 and h_avg > fail_cutoff or ind == 1 and h_avg < fail_cutoff): 
                    result += '&  $  \\textcolor{{fail}}{{{:.1f}\\pm{:.1f}}} $  '.format(h_avg, h_std)
                else:
                    result += '&  $    {:.1f}\\pm{:.1f} $  '.format(h_avg, h_std)

        result += '\\\\ \n'
    with open('results/table.tex', 'a') as f:
        f.write(result)