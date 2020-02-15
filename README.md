# adv-atk-neu

## Getting started
The class ParticleClassifier is written to aid in all aspects: from preparation of data to evaluating the attack on the model. x and y represent the pre-processed images and labels respectively. Each data set has 3 versions: train, test and attacked. If no data sets are explicitly passed into them, most functions would execute into nMost functions return self so functions can be chained. An example of building a model and evaluating an attack is shown:

```
classifier = ParticleClassifier()\ # Instantiates the class
            .load_data('data/toy_data.csv')\ # Saves data into self.images and self.labels
            .train_test_split(test_size=0.2)\ # Splits images and labels into self.images_train and self.images_test
            .pre_proc_images(train=True, test=True)\
            .one_hot_encode_labels(train=True, test=True)\
            .train_model()\
            .evaluate_model()\
```

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
