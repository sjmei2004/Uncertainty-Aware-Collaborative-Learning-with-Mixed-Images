# Uncertainty-Aware-Collaborative-Learning-with-Mixed-Images

This repository contains the code implementation for the paper "Uncertainty-Aware Collaborative Learning with Mixed Images for Semi-supervised Medical Image Segmentation".

## Requirements

List of dependencies required to run the code.

- python
- numpy
- torch
- h5py
- nibabel
- scipy
- skimage
- tqdm
- medpy

These dependencies can be installed using the following command:

    pip install -r requirements.txt

## Dataset

The dataset used for this project can be downloaded from the following links:

- https://promise12.grand-challenge.org/
- https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

## How to train

1. Clone the repository
2. Install the required packages using the command mentioned above
3. Download the dataset from the link provided above and extract it to the data/ directory
To train a model,
```
python train_Promise12.py  #for Prostate training
python train_ACDC.py  #for ACDC training
```
To test a model,
```
test_2D_Promise12.py  #for Prostate testing
python test_2D_ACDC.py  #for ACDC testing
```
:beers: The implementation of other SSL approaches can be referred to the author Dr. Luo's [SSL4MIS project](https://github.com/HiLab-git/SSL4MIS).
