# PointNet Model for 3d point cloud object classification

This program is part of Deep Learning Course - CSL7590 at IIT Jodhpur

## Problem Statement
Design and implement a neural network architecture to solve the problem of object classification for
the categories available in ModelNet10 dataset(http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip).

## Directory Structure

- project_root/
  - data/
    - bathtub
      - test
        - 001.off
        - 002.off
      - train
        - 001.off
        - 002.off
    - chair
      - test
        - 001.off
        - 002.off
      - train
        - 001.off
        - 002.off
  - model/
    - PointNetSequential.py
    - PointNet.py
  - output/
    - model_architecture.png
    - model_weights.pth
  - utils
   - data_preprocessing.py
  - eval.py
  - train.py

1. `model/PointNetSequential.py` contains the model architecture implementation.
2. `eval.py` is the evaluation script to classify the object from test set.
3. `train.py` trains the model on ModelNet10 dataset.


## How to Run?

#### Setup Environment

1. Setup environment. Preferrably use anaconda.
    ```
    conda env create -f environment.yml
    ```
2. Activate Conda environment
    ```
    conda activate deep_learning
    ```
3. Create an empty folder with names data and output.
4. Download the dataset and extract the contents in the data folder. Refer to directory structure for more details


#### Run Scripts

1. Training

```
python train.py
```

The model gets stored inside the output directory as model_weights.pth

2. Eval - This will classify the objects in test set

```
python eval.py
```

This will use pretrained model inside output directory - model_weights.pth