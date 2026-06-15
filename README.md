# 1project-ITCS

# Iris Classification Project

## Overview

This project trains a machine learning model on the Iris dataset using Scikit-Learn.

The model is trained, evaluated, saved as a Joblib file, and a confusion matrix is generated.

## Project Structure

```text
.
├── notebooks/
│   └── iris_model.ipynb
├── outputs/
│   ├── model.joblib
│   └── confusion_matrix.png
├── src/
│   └── train.py
├── requirements.txt
└── README.md
```

## Installation

Install the required packages:

matplotlib
joblib
os
scikit-learn
pandas
numpy


## Run the Project

Execute the training script:

 src/train.py


## Outputs

Running the script generates:

- `outputs/model.joblib` – saved trained model
- `outputs/confusion_matrix.png` – confusion matrix image

## Model

- Dataset: Iris Dataset
- Algorithm: Decision Tree Classifier
- Evaluation: Accuracy Score, Precision, and Confusion Matrix
