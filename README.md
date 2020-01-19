# adv-atk-neu

## Introduction
This project explores adverserial attacks on a convolutional neural network classifier built to identify particles from particle detector outputs.

## Data
Data is given in a csv file with each line containing the class followed by a 50 x 50 image. 3 classes are present: 0 for no particle, 1 for electron, and 2 for muon. Background noise is of a Gaussian nature.

## Model
The CNN built was able to achieve an accuray of around 0.98. Precision and recall values were above 0.96 for each class.

## Attacks
The following attacks have been attempted:
- Single Pixel Attack:
  - Dead Channel: Value of pixel set to 0
  - Hot Channel: Value of pixel set to 2 times of max of image

## Notebooks
EDA.ipnyb: Shawn
First Attack.ipnyb: Suhasan
