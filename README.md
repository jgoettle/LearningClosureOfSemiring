<div align="center">    
 
# Learning DAGs from Data with Few Root Causes

</div>
 
## Description   
This is the implementation of the method and the experiments described in my bachelor thesis.
To use it you can simply install Python with the required packages in [`requirements.txt`](requirements.txt)

### Method description
In the thesis a new data generation process is introduced, which is derived from the linear SEM, where the transitive closure is computed over an (idempotent) semiring. 
This changes the meaning of the overall influence between nodes to for example the shortest path or the maximal reliabilty. 

These influences are then learned by using [SparceRC]https://github.com/FenTechSolutions/CausalDiscoveryToolbox) and computing the plus-times transitive closure of the infered matrix.

Additionally it introduces the weighted transitive reduction, which is also learned by my method.

A demonstration is shown in [`DAG_learning_semirings_demo.ipynb`](DAG_learning_semirings_demo.ipynb)




## Full experiment reproducibility 
Current version of the code was run successfully in  with python 3.9.7.

### Environment and packages
Setup the python environment:

```bash
# create a new conda environment
conda create -n learnDagsSemiring  python=3.12.8
conda activate learnDagsSemiring

# install all requirements with pip
pip install -r requirements.txt
```

### Other


Make sure to have R installed and that the path to the R directory in file **experiment.py** is correct. Also note the required packages on R for the [causal discovery toolbox](https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html).

<!-- ```bash
R version 4.2.1
install.packages("BiocManager")
BiocManager::install(c("graph", "RBGL"))

install.packages("pcalg")

install.packages(pkgs=pckg, type="source", repos=NULL)
url <- 'https://cran.r-project.org/src/contrib/Archive/SID/SID_1.0.tar.gz'
pckg <- 'SID_1.0.tar.gz'
download.file(url = url, destfile = pckg)
install.packages(pkgs=pckg, type="source", repos=NULL)
library('SID')

q()
``` -->

### Execution   
```bash
# run all experiments
bash run.sh
# plots and tables
bash report.sh
```