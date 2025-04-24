
# NeuralVision: Brain Tumour Detection using Deep Learning

This project utilizes Convolutional Neural Networks (CNNs) to automate the detection and classification of brain tumours from MRI scans. The goal is to support medical diagnosis by identifying tumour presence and type with high accuracy using AI-driven image analysis.

## Project Overview

- **Goal**: Classify MRI images into tumour types (e.g., glioma, meningioma, pituitary) or identify absence of tumour.
- **Dataset**: MRI images sourced from publicly available datasets.
- **Output**: Multi-class classification (Tumour Type or No Tumour).

## Deep Learning Architecture

- **Preprocessing**:
  - Image resizing, normalization, and augmentation.
- **Model Architectures**:
  - Transfer learning using models like VGG16, InceptionV3, ResNet50
  - Custom CNN designed for image classification
- **Training**:
  - Optimized using Adam optimizer and categorical cross-entropy loss.
  - Validation using stratified k-fold cross-validation.
- **Evaluation**:
  - Accuracy, precision, recall, F1-score
  - Confusion matrix and ROC curves

## Results Summary

| Model         | Training Accuracy | Validation Accuracy |
|---------------|-------------------|---------------------|
| VGG16         | 99.86%            | 98.82%              |
| ResNet50      | 99.98%            | 98.95%              |
| InceptionV3   | 98.86%            | 96.57%              |
| Custom CNN    | 98.98%            | 99.10%              |


## Tools & Libraries

- Python, TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Scikit-learn

---

