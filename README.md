##COMPS492F Machine Learning 


# Shoe Brands Classification Project

## Introduction
This project aims to implement shoe brand classification using machine learning techniques, specifically transfer learning using the YOLOv8 (You Only Look Once) model.

## Techniques Selection
Initial experiments with a CNN-based untrained model approach were conducted, but the performance was underwhelming. The model achieved a maximum accuracy of 65% on the testing set before overfitting. This suboptimal performance led to exploring alternative methods, ultimately settling on transfer learning using YOLOv8.

## Data Preparation
The provided data includes a training set and a testing set, of three classes (shoe brands). All images are in 240x240 resolution.

## Data Preprocessing and Training
The project uses the YOLOv8 model, which is based on PyTorch. The Ultralytics dependency is required.

The training was done on Google Colab. The training script (train.ipynb) is provided in the submitted files. The default classification model of YOLOv8 is used for transfer learning.

YOLOv8's default training method automatically preprocesses the training set and applies data augmentation techniques like random scaling and mosaic. The model is trained for 30 epochs, reaching an accuracy of 94.7% on the testing set. The trained weights with the best accuracy are automatically saved as best.pt, which can be downloaded or evaluated.

## Testing and Validation
A test script (test_model.py) is also provided in the submitted files. The script can receive a testing folder as input, apply the trained model, and save the classification results to a .txt file in the designated path.

The testing script can also automatically check the overall accuracy of the testing data if the testing folder contains folders with class names (e.g., nike, adidas, converse).

The validation of the model is done using 4 images downloaded from the Nike official website.

## Usage
To run the testing script, use the following command:

```
test_model.py [test data path] [trained model path] [saved result path]
```

For example:

```
test_model.py "data/test" "best.pt" "results/results.txt"
```

This will apply the trained model to the test data, save the results to the "results/results.txt" file, and calculate the overall accuracy if the test data is organized by class folders.
