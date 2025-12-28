# omnilib-ml

Official repostitory for the paper Antibody folding fitness encoded in complementarity determining region sequences

Code to reproduce the models and figures will be provided, and pretrained model checkpoints are located here.

In general, these models work best with the synthetc libraries described in this paper: [Yeast surface display platform for rapid discovery of conformationally selective nanobodies]( https://www.nature.com/articles/s41594-018-0028-6), as well as the current work.

## Installation

Required packages are listed in the requirements.txt file.
To install this repository, run the following command:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data

We provide preprocessed and generated data for model training, and evaluation. Data generated as part of the
figures in the paper are also provided.
The data folder is stored at the following zenodo link: [TODO: Add link]

Once downloaded, the data can be extracted to the data folder in the root of the repository.

```bash
tar -xvf data.tar.gz
```

## Models


### Classifier Models

The classifier models require a dataframe with the following columns:

- Omnilib: cdr1, cdr2, cdr3, stability columns


## Model Checkpoints

As part of this work, we provide the following pretrained models, as well as code to train new models, and run inference.

### OmniLib Fitness Classifiers

| Model | Description | Parameters | Link |
| --- | --- | --- | --- |
|CNN| Convolutional Neural Network| 8k| Link|
|LR| Logistic Regression| 1k| Link|

### Figures

All figures from the paper that use the datasets and models in this repository are provided in the `figures` directory.
For each figure, there is a juptyer notebook that generates the figure. The notebooks are named according to the figure they generate.
Furthermore, the outputs of the notebooks are provided in the `figures` directory as well.
