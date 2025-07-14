# Image Classification with Tiny ImageNet

This project classifies images using machine learning models on a selected subset of Tiny ImageNet. The focus is on classic models (Logistic Regression, KNN, SVM).

## Dataset

- 10 selected categories from Tiny ImageNet Kaggle.
- The classes: acorn, pizza, espresso, orange, pomegranate, bison, goldfish, police truck, cauliflower, coral reef.
- 5000 training images, 500 test images.
- All images resized to 300x300.
- Features extracted with pretrained ResNet18.
- Feature scaling applied to all feature vectors.

## Models Used

- Logistic Regression (Multinomial)
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

## Model Training & Experiments

### Logistic Regression

- Used `train_test_split` with 4 different test sizes: `0.1`, `0.2`, `0.3`, and `0.4`.
- Solver: `L-BFGS`, max_iter: `1000`.
- Evaluated accuracy, precision, recall, and F1-score at each split.

**Best Accuracy:** `89.00%` (test size = 0.2)

---

### K-Nearest Neighbors

- Used `GridSearchCV` to find the best value of `k`.
- Tested values: `k = 2, 4, 6, 8` with GridSearchCV.
- Also manually tested `k = 6, 7, 8, 9` to compare accuracy and sensitivity.

**Best Accuracy:** `81.40%` (k = 7)

---

### Support Vector Machine

- Used `GridSearchCV` to tune regularization parameter `C` for each kernel.
- Kernels tested: `linear`, `rbf`, `poly`, `sigmoid`.
- C values tested: `0.1`, `0.5`, `1.0`, `1.5`.

**Best Accuracy:** `90.80%` (RBF kernel, C = 1.5)

---

## Evaluation Metrics

All models were evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

## Dataset Source

- [Tiny ImageNet Dataset on Kaggle](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)

