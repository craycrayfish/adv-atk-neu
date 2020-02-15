# adv-atk-neu

## Getting started
The class ParticleClassifier is written to aid in all aspects: from preparation of data to evaluating the attack on the model. x and y represent the pre-processed images and labels respectively. Each data set has 3 versions: train, test and attacked. If no data sets are explicitly passed into them, most functions would execute into nMost functions return self so functions can be chained. An example of building a model and evaluating an attack is shown:

```
classifier = ParticleClassifier()\ 
            # Instantiates the class
            .load_data('data/toy_data.csv')\ 
            # Saves data into self.images and self.labels
            .train_test_split(test_size=0.2)\ 
            # Splits images and labels into self.images_train and self.images_test
            .pre_proc_images(train=True, test=True, attacked=False, filters=False, rescale=True)\
            # Applies rescaling but not filtering pre-processing to train and test image sets only
            .one_hot_encode_labels(train=True, test=True)\
            # Applies one hot encoding to train and test label sets
            .train_model()\
            # Train the model
            .evaluate_model()\
            # Evaluate model in terms of precision and recall
            .apply_attack(classifier.add_hot_area, size=(2,2), value=value)\
            # Applies attack, given as a function, to images (default test). Further arguments are passed on
            # to attack function. E.g. classifier.add_hot_area(images, size=(2,2), value=value) in this case.
            # Attack function should return images in the same format after executing attack on them.
            .pre_proc_images(attacked=True, filters=False)\
            # Pre-process the attacked images
            .one_hot_encode_labels(attacked=True)\
            # One hot encode the labels of attacked images
            .evaluate_attack()
            # Get precision and recall values for attacked predictions against non-attacked predictions.
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
- Background Neutrons:
  - Single Low-Energy Neutron: Value of a group of 4 pixels set to between mean and max

## Notebooks
EDA.ipnyb: Shawn

First Attack.ipnyb: Suhasan
