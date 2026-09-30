# Breast Cancer Detection

A small machine learning and deep learning project for breast cancer classification.

The project started with classical machine learning models and was later extended to image classification using a CNN and transfer learning.

## Project Structure

This repository contains two parts:

- `breast-cancer-ml/` — classical machine learning
- `breast-cancer-CNN/` — image classification with deep learning

---

## 1. Classical Machine Learning

The first part uses tabular breast cancer data.

### Preprocessing

The dataset was prepared using:

- Removing unnecessary columns
- Handling missing values
- Label encoding
- Feature scaling with `StandardScaler`
- PCA for dimensionality reduction

### Models

Two models were tested:

- Logistic Regression
- Support Vector Classifier (SVC)

### Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve

PCA and t-SNE were also used to get a better view of the data.

---

## 2. Breast Cancer CNN

The second part of the project focuses on image classification.

The goal is to classify images into two classes:

- Benign
- Malignant

### Dataset

The images are divided into:

```text
data/
├── train/
├── valid/
└── test/
```
The dataset is not perfectly balanced, so the class distribution was checked before training.

### Preprocessing

The images were resized and normalized before being passed to the model.

Training images also use data augmentation such as:

Random horizontal flip
Random resized crop
Normalization
Model

For the final experiment, transfer learning was used with a pretrained ResNet18.

The pretrained layers were frozen and the final classification layer was replaced for the two-class problem.

The final setup used:

Image size: 384 × 384
Batch size: 32
Optimizer: Adam
Weight decay: 1e-2
Different learning rates for the pretrained layers and classifier
Early stopping
Best model selected using validation loss

### Evaluation

The CNN model was evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Training / Validation Loss

The generated plots are available in the img/ directory.

What I Learned

This project was mainly about understanding the full machine learning workflow rather than trying to get the highest possible accuracy.

Some of the main things I worked with were:

Data preprocessing
Train / validation / test splits
Feature scaling
PCA
Classical ML models
CNN architecture
Transfer learning
Data augmentation
Overfitting
Regularization
Learning rate
Weight decay
Model evaluation

One of the main challenges in the CNN part was overfitting. Increasing model complexity did not always improve validation performance, so the experiments focused more on finding a stable training setup.

Notes

This is an educational project and is not intended for medical diagnosis or clinical use.

The results should not be interpreted as medical performance or as a replacement for professional medical evaluation.
