from neurips_experiments.data import data_generation
import neurips_experiments.utils
import neurips_experiments.evaluation.utils
import neurips_experiments.plot_experiment
from sparserc.sparserc import sparserc_solver
import transitive_closure.transitive_closure 

from tqdm import tqdm
import time
import pandas as pd
import os
import numpy as np

# varsortability
import transitive_closure.utils

# Causal Discovery toolbox + setup R path
import cdt 
cdt.SETTINGS.rpath = '/opt/homebrew/bin/Rscript'  #'C:/Program Files/R/R-4.2.1/bin/Rscript'



if __name__ == '__main__':
    parser, args = neurips_experiments.utils.get_args()
    print(vars(args))

    # naming the output files according to the experimental settings
    filename, label = neurips_experiments.utils.get_filename(parser, args)

    # make directory to put results
    if not os.path.exists("results/{}/".format(filename)):
        os.makedirs("results/{}/".format(filename))

    # make sub-directories to put results
    for semiring in args.semirings:  
        if not os.path.exists("results/{}/{}/".format(filename, semiring )):
            os.makedirs("results/{}/{}/".format(filename, semiring))

    for n in args.samples:
        for d in args.nodes:
            print('results/{}.csv'.format(filename))
            with open('results/{}.csv'.format(filename), 'a') as f:
                f.write('{}\n'.format(label))

                print('samples = {}, nodes = {}, noise = {}'.format(n, d, args.noise))
                f.write('samples = {}, nodes = {}, noise = {}\n'.format(n, d, args.noise))

                current = {}
                hist_bins = {}

                for key in args.semirings:
                    current[key] = []

                # avg_cond_num = 0 
                # avg_varsortability = 0
     
                for r in tqdm(range(args.runs)):
                    
                    (a, b) = tuple(args.weight_bounds)
                    k = args.edges
                    
                    B_true = data_generation.simulate_dag(d, k * d, args.graph_type)
                    W_help = data_generation.simulate_parameter(B_true, w_ranges=((-b,-a), (a, b))) # sampling uniformly the weights

                    for semiring in args.semirings:  
                
                        # graph initialization
                        start = time.time()
                        if semiring == 'plus-times':
                            W_true = W_help
                        else:
                            W_true = abs(W_help)

                        X, C_true, W_tc_true = data_generation.sparse_rc_sem(W_true, n, sparsity=args.sparsity, std=args.noise_std, 
                                                    noise_type=args.noise, noise_effect=args.noise_effect, semiring=semiring)

                        df = pd.DataFrame(C_true)
                        df.to_csv('results/{}/{}/C_run_{}_{}_{}.csv'.format(filename, semiring, r, d, n), header=None, index=False)
                        df = pd.DataFrame(X)
                        df.to_csv('results/{}/{}/X_run_{}_{}_{}.csv'.format(filename, semiring, r, d, n), header=None, index=False)
                        df = pd.DataFrame(W_true)
                        df.to_csv('results/{}/{}/W_true_run_{}_{}_{}.csv'.format(filename, semiring, r, d, n), header=None, index=False)
                        df = pd.DataFrame(W_tc_true)
                        df.to_csv('results/{}/{}/W_tc_true_run_{}_{}_{}.csv'.format(filename, semiring, r, d, n), header=None, index=False)

                        print("\n\nData generation process done. Time: {:.3f}\n\n".format(time.time() - start))

                        start = time.time()
                        W_pt_est = sparserc_solver(X, lambda1=0, lambda2=1, epochs=args.sparserc_epochs, omega=args.omega)
                        T = time.time() - start
                        W_pt_est, edges_rem = transitive_closure.utils.remove_cyclic_edges_by_weight(W_pt_est)


                        W_tc_est = transitive_closure.transitive_closure.transitive_closure_dag(W_pt_est, semiring= "plus-times")
                        
                        B_tc_est = W_tc_est != 0
                        B_tc_true = W_tc_true != 0 
                        
                        # save result for future reference
                        df = pd.DataFrame(W_pt_est)
                        df.to_csv('results/{}/{}/W_pt_est_run_{}_{}_{}.csv'.format(filename, semiring, r, d, n), header=None, index=False)
                        df = pd.DataFrame(W_tc_est)
                        df.to_csv('results/{}/{}/W_tc_est_run_{}_{}_{}.csv'.format(filename, semiring, r, d, n), header=None, index=False)


                        # Create subplot
                        neurips_experiments.plot_experiment.visualize(B_tc_true, B_tc_est, "tc_" + semiring, filename)


                        if args.eval_mode == "closure":
                            neurips_experiments.evaluation.utils.compute_metrics(semiring, current, hist_bins, edges_rem, r, T, B_tc_true, W_tc_true, B_tc_est, W_tc_est, args)

                        elif args.eval_mode == "reduction":
                            W_tr_true = transitive_closure.transitive_closure.transitive_reduction_weighted(W_tc_true, semiring)
                            W_tr_est = transitive_closure.transitive_closure.transitive_reduction_weighted_with_correction(W_tc_est, semiring, correction=args.cor, correction_val=args.cor_val)
                            B_tr_est = W_tr_est != 0 
                            B_tr_true = W_tr_true != 0
                            neurips_experiments.plot_experiment.visualize(B_tr_true, B_tr_est, "tr_" + semiring, filename)
                            df = pd.DataFrame(W_tr_est)
                            df.to_csv('results/{}/{}/W_tr_est_run_{}_{}_{}.csv'.format(filename, semiring, r, d, n), header=None, index=False)

                            neurips_experiments.evaluation.utils.compute_metrics(semiring, current, hist_bins, edges_rem, r, T, B_tr_true, W_tr_true, B_tr_est, W_tr_est, args)

                        elif args.eval_mode == "closure-of-reduction":
                            if args.cor != "orig":
                                W_tr_true = transitive_closure.transitive_closure.transitive_reduction_weighted(W_tc_true, semiring)
                                W_tr_est = transitive_closure.transitive_closure.transitive_reduction_weighted_with_correction(W_tc_est, semiring, correction=args.cor, correction_val=args.cor_val)

                                W_tc_tr_est = transitive_closure.transitive_closure.transitive_closure_dag(W_tr_est,  semiring)
                                B_tc_tr_est = W_tc_tr_est != 0

                                df = pd.DataFrame(W_tc_tr_est)
                                df.to_csv('results/{}/{}/W_tc_tr_est_run_{}_{}_{}.csv'.format(filename, semiring, r, d, n), header=None, index=False)

                                neurips_experiments.evaluation.utils.compute_metrics(semiring, current, hist_bins, edges_rem, r, T, B_tc_true, W_tc_true, B_tc_tr_est, W_tc_tr_est, args)
                            else:
                                neurips_experiments.evaluation.utils.compute_metrics(semiring, current, hist_bins, edges_rem, r, T, B_tc_true, W_tc_true, B_tc_est, W_tc_est, args)

                        elif args.eval_mode == "plus-times-original":
                            W_pt_true=transitive_closure.transitive_closure.transitive_reduction_weighted(W_tc_true, semiring="plus-times")
                            B_pt_est = W_pt_est != 0
                            B_pt_true = W_pt_true != 0

                            neurips_experiments.evaluation.utils.compute_metrics(semiring, current, hist_bins, edges_rem, r, T, B_pt_true, W_pt_true, B_pt_est,W_pt_est, args)
                        
                        elif args.eval_mode == "root-causes":
                           neurips_experiments.evaluation.utils.compute_metrics_rc(semiring, current, X, W_pt_est, C_true, args)
                        else: 
                            raise ValueError("unknown evaluation mode")

                # save average results in csv
                neurips_experiments.evaluation.utils.save_results(current, f, hist_bins, filename, args)
