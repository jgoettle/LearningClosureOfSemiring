#!/bin/bash
# All tables and plots from the thesis

# Try to empty the result folder and rerun run.sh if the script does not work.


# python neurips_experiments/print_table_tex.py --nodes 20

# python neurips_experiments/print_table_tex.py

# python neurips_experiments/plot_experiment.py --nodes 20 50 100 200 300
# python  neurips_experiments/plot_experiment.py --nodes 20 50 100 200 300 --legend True

python neurips_experiments/plot_experiment.py --semirings 'min-plus' 'max-times' --hist 'True' --eval_mode 'plus-times-original'
# python neurips_experiments/plot_experiment.py --semirings 'min-plus' 'max-times' --hist 'True' --hist_bound 1 5 --eval_mode 'closure'

# python  neurips_experiments/plot_experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'reduction'
# python  neurips_experiments/plot_experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'reduction' --legend True

# python  neurips_experiments/plot_experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'closure-of-reduction'
# python  neurips_experiments/plot_experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'closure-of-reduction' --legend True