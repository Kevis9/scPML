# scPML: Pathway-based Multi-view Learning for Cell Type Annotation from Single-cell RNA-seq Data
A PyTorch implementation of scPML.
[![DOI](https://zenodo.org/badge/446746824.svg)](https://zenodo.org/doi/10.5281/zenodo.10155115)

## Requirements
* R >= 4.2.1
* python >= 3.8.12
### Python package version
* scikit-learn >= 1.0.2
* numpy >= 1.23.3
* pytorch >= 1.10.2
* torch-geometric >= 2.0.3
* networkx >= 2.8.4
* pandas >= 1.4.3
* scipy >= 1.9.1
We recommend upgrading all packages to the latest version.


## Input Data
When using your own data, you have to provide:
* the raw data matrix of training data and cells labels.
* the raw data matrix of test data.
Training and test data should share the same gene features.

For the sake of convenience, we use a directory tree to organize experiments and data:
```
-- proj_name
  -- raw_data
    -- ref
        data_1.csv
        label_1.csv
    -- query
        data_1.csv
  -- data
    -- ref
    -- query       
```
When using your own data, put raw training data matrix in the `raw_data/ref/` and name it `data_1.csv`, raw training label in `raw_data/ref/` and name 
it `label_1.csv`. Put the raw test data in the `raw_data/query/` directory.

The `data` directory contains the pre-processed data. More details can be seen in the demo directory.

## Run the demo

### Similarity matrix construction
```
    cd demo
    Rscript ..\utils\get_sm.R seq_well_10x_v3    
```
### Data preprocess
```        
    Rscript ..\utils\pre_process.R seq_well_10x_v3
    python ..\utils\data_csv2h5.py --path=seq_well_10x_v3 --subpath=raw_data
    python ..\utils\data_csv2h5.py --path=seq_well_10x_v3 --subpath=data    
```
### Run scPML
```
    python main.py
```

## Output
The results will be stored in the `result` folder.






