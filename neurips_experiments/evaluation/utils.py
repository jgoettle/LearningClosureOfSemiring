import numpy as np
import neurips_experiments.evaluation.evaluation 
import neurips_experiments.utils
import pandas as pd
import cdt

def compute_metrics(semiring, current, hist_bins, edges_rem, r, T, B_true, W_true, B_est, W_est, args):
    acc = neurips_experiments.evaluation.evaluation.count_accuracy(B_true, B_est)
    nmse = np.linalg.norm(W_est - W_true) / np.linalg.norm(W_true)

    # R needed
    try: # sid computation assumes acyclic graph
        if(not  neurips_experiments.utils.is_dag(B_est)):
            print("Warning, output is not a DAG, SID doesn't make sense")
        sid = neurips_experiments.utils.timeout(timeout=1000)(cdt.metrics.SID)(B_true, B_est) 
    except:
        sid = float("nan")

    current[semiring].append([acc['shd'],  acc['tpr'], sid, acc['fpr'],acc['nnz'], nmse, edges_rem, T])
    print("Results, SHD, TPR, SID, FPR, NNZ, NMSE, Edges RM, T")
    print("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}"
            .format(semiring, current[semiring][r][0], current[semiring][r][1], current[semiring][r][2], current[semiring][r][3], current[semiring][r][4], current[semiring][r][5], current[semiring][r][6], current[semiring][r][7]))
    
    if args.hist == "True":
        all_entries = W_true.flatten()

        # Define the number of bins for the range [0, 1]
        num_bins = args.hist_bins
        min_val = -args.hist_bound[0]
        max_val = args.hist_bound[1]
        

        # Compute the histogram (density normalized to sum to 1)
        hist, _ = np.histogram(all_entries[all_entries != 0], num_bins, (min_val, max_val))
        
        if semiring not in hist_bins:
            # Initialize with the current histogram
            hist_bins[semiring] = hist
        else:
            # Elementwise addition: both arrays must have the same shape
            hist_bins[semiring] += hist

def compute_metrics_rc(semiring, current, r, X, W_pt_est, C_true):

    c_nmse, c_tpr, c_fpr = neurips_experiments.evaluation.evaluation.rc_approximation(semiring, X, W_pt_est, C_true)

    current[semiring].append([c_tpr, c_fpr, c_nmse])
    print("Results, C_TPR, C_FPR, C_NMSE, T")
    print("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}"
            .format(semiring, current[semiring][r][0], current[semiring][r][1], current[semiring][r][2], current[semiring][r][3]))
    

def save_results(current, f, hist_bins, filename, args):      
    # Log results
    avg = {}
    std = {}
    print("Evaluation mode: {}\n".format(args.eval_mode))
    if args.eval_mode == "root-causes":
        f.write("Results, C_TPR, C_FPR, C_NMSE ,T \n")
        for semiring in args.semirings:
            avg[semiring] = np.mean(current[semiring], axis=0)
            std[semiring] = np.std(current[semiring], axis=0)
            
            f.write("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(semiring, avg[semiring][0], avg[semiring][1], avg[semiring][2], avg[semiring][3]))
            f.write("Std {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(semiring, std[semiring][0], std[semiring][1], std[semiring][2], std[semiring][3]))
            print("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(semiring, avg[semiring][0], avg[semiring][1], avg[semiring][2], avg[semiring][3]))
            print("Std {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(semiring, std[semiring][0], std[semiring][1], std[semiring][2], std[semiring][3]))

    else:
        f.write("Results, SHD, TPR, SID, FPR, NNZ, NMSE, E_RM, T \n")

        for semiring in args.semirings:
            if args.hist == "True":
                # save histo
                df = pd.DataFrame(hist_bins[semiring] / args.runs)
                df.to_csv('results/hist_{}_{}.csv'.format(semiring, filename), header=None, index=False)
            else:
                avg[semiring] = np.mean(current[semiring], axis=0)
                std[semiring] = np.std(current[semiring], axis=0)
                
                f.write("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(semiring, avg[semiring][0], avg[semiring][1], avg[semiring][2], avg[semiring][3], avg[semiring][4], avg[semiring][5], avg[semiring][6], avg[semiring][7]))
                f.write("Std {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(semiring, std[semiring][0], std[semiring][1], std[semiring][2], std[semiring][3], std[semiring][4], std[semiring][5], std[semiring][6], std[semiring][7]))
                print("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(semiring, avg[semiring][0], avg[semiring][1], avg[semiring][2], avg[semiring][3], avg[semiring][4], avg[semiring][5], avg[semiring][6], avg[semiring][7]))
                print("Std {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(semiring, std[semiring][0], std[semiring][1], std[semiring][2], std[semiring][3], std[semiring][4], std[semiring][5], std[semiring][6], std[semiring][7]))





#TODO remove 
def compute_metrics_old(semiring, current, filename, r, T, X, C_true, B_true, B_tc_true, W_true, W_tc_true, B_tr_true, B_est, B_tc_est, W_est, W_tc_est, B_tr_est, args):
    d = X.shape[1]
    c_nmse, c_tpr, c_fpr = neurips_experiments.evaluation.evaluation.rc_approximation(args.method, semiring, X, W_est, C_true)
    nmse = np.linalg.norm(W_est - W_true) / np.linalg.norm(W_true)
    tc_nmse = np.linalg.norm(W_tc_est - W_tc_true) / np.linalg.norm(W_tc_true)
    acc = neurips_experiments.evaluation.evaluation.count_accuracy(B_true, B_est)
    tc_acc = neurips_experiments.evaluation.evaluation.count_accuracy(B_tc_true , B_tc_est)
    tr_acc = neurips_experiments.evaluation.evaluation.count_accuracy(B_tr_true , B_tr_est)
    '''
    R needed
    try: # sid computation assumes acyclic graph
        if(not  neurips_experiments.utils.is_dag(B_est)):
            print("Warning, output is not a DAG, SID doesn't make sense")
        print("tstsasdf")
        sid = neurips_experiments.utils.timeout(timeout=100)(cdt.metrics.SID)(B_true, B_est) 
        print("tstsasdf")
    except:
        sid = float("nan")
    current[method].append([shd, acc['tpr'], acc['nnz'], sid, nmse, c_tpr, T, c_nmse, c_fpr, acc['fpr']])
    '''
    current[semiring].append([acc['shd'], acc['tpr'], acc['nnz'], nmse, c_tpr, T, c_nmse, c_fpr, acc['fpr'], tc_acc['shd'], tc_acc['tpr'], tc_acc['nnz'], tc_nmse, tc_acc['fpr'], tr_acc['shd'], tr_acc['tpr'], tr_acc['nnz'], tr_acc['fpr']])
    print("Results, SHD, TPR, NNZ, NMSE, C_TPR, T, C_NMSE, C_FPR, FPR, TC_SHD, TC_NNZ, TC_NMSE, TC_TPR, TC_FPR, TR_SHD, TR_NNZ, TR_TPR, TR_FPR")
    print("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}"
            .format(semiring, current[semiring][r][0], current[semiring][r][1], current[semiring][r][2], current[semiring][r][3], current[semiring][r][4], current[semiring][r][5], current[semiring][r][6], current[semiring][r][7], current[semiring][r][8], current[semiring][r][9], current[semiring][r][10], current[semiring][r][11], current[semiring][r][12], current[semiring][r][13], current[semiring][r][14], current[semiring][r][15], current[semiring][r][16], current[semiring][r][17]))
    # looking at weights
    if d > 100:
        df = pd.DataFrame(W_est)
        df.to_csv('results/W_est_{}_nodes_{}_{}.csv'.format(filename, d, semiring), header=None, index=False)
        df = pd.DataFrame(W_true)
        df.to_csv('results/W_true_{}_nodes_{}_{}.csv'.format(filename, d, semiring), header=None, index=False)