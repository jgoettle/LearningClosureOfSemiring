#!/bin/bash
# All experiments

python  experiment.py --nodes 20
python  experiment.py 

python experiment.py --nodes 20 50 100 200 300

python experiment.py  --semirings 'min-plus' 'max-times' --hist 'True' --eval_mode 'plus-times-original'

python experiment.py --semirings 'min-plus' 'max-times' --hist 'True' --hist_bound 1 5 --eval_mode 'closure'

python experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'reduction'  --cor_val 0.0
python experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'reduction'  --cor_val 0.1
python experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'reduction'  --cor_val 0.2
python experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'reduction'  --cor_val 0.1 --cor 'minus'
python experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'reduction'  --cor_val 0.2 --cor 'minus'
python experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'reduction'  --cor_val 0.8 --cor 'sr-t'
python experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'reduction'  --cor_val 0.9 --cor 'sr-t'
python experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'reduction'  --cor_val 1.2 --cor 'sr-t'
python experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'reduction'  --cor_val 1.4 --cor 'sr-t'

python experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'closure-of-reduction'  --cor 'orig'
python experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'closure-of-reduction'  --cor_val 0.1
python experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'closure-of-reduction'  --cor_val 0.2
python experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'closure-of-reduction'  --cor_val 1.2 --cor 'sr-t'
python experiment.py --nodes 20 40 60 80 100 --semirings 'max-times' --eval_mode 'closure-of-reduction'  --cor_val 1.4 --cor 'sr-t'