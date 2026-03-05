# Breast Cancer Classification Using Scikit-Learn

## Project Purpose

The purpose of this project is to apply **machine learning classification techniques** to a real-world medical dataset. Using the built-in **Breast Cancer dataset from Scikit-Learn**, this program builds and evaluates multiple classification models to determine which performs best at predicting whether a tumor is **benign or malignant**.

Early detection plays a major role in reducing breast cancer mortality rates. Machine learning models can assist medical professionals by identifying patterns in patient data and helping classify potential cancer cases more accurately.

This assignment focuses on:
- Importing and preparing real-world data
- Training multiple machine learning models
- Evaluating model performance using statistical metrics
- Comparing results to determine the best classifier

Only the allowed libraries were used:
- `scikit-learn`
- `numpy`
- `pandas`
- `scipy`
- standard Python modules
---
## Dataset

The program uses the **Breast Cancer Wisconsin dataset** included in Scikit-Learn.

Dataset characteristics:

- **569 samples**
- **30 numerical features**
- **2 classifications**
  - Malignant (0)
  - Benign (1)

Example features include:
- Mean radius
- Mean texture
- Mean perimeter
- Mean area
- Mean smoothness
The dataset is split using the standard **80/20 training and testing division**.
---
## Class Design and Implementation

The program is organized using a class-based structure to improve readability, maintainability, and modularity. The main class handles dataset loading, preprocessing, model training, and evaluation.

### Main Class

`BreastCancerClassifier`

This class encapsulates all logic for:
- loading and preparing the dataset
- training machine learning models
- evaluating model performance
- comparing results across models
---
## Class Attributes
`data`
- Stores the dataset features and labels loaded from Scikit-Learn.
`X`
- The feature matrix containing the 30 medical measurement variables.
`y`
- The target variable indicating whether the tumor is benign or malignant.
`X_train`, `X_test`
- Training and testing feature sets created from the dataset split.
`y_train`, `y_test`
- Training and testing labels.
`models`
- A collection of the classification models used in the project.
`results`
- Stores performance metrics for each trained model.
---

## Class Methods
`load_data()`
- Loads the breast cancer dataset from Scikit-Learn.
- Extracts the feature matrix and target labels.
`prepare_data()`
- Splits the dataset into training and testing sets using an 80/20 split.
- Applies feature normalization using **StandardScaler** to ensure fair comparison between models.
`train_models()`
- Trains three different classification models:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
`evaluate_models()`
- Evaluates each model using several performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
`compare_models()`
- Compares the performance metrics of all models to determine the best-performing classifier.
---
## Model Performance Summary

Three classification models were tested:

### Logistic Regression (Best Performing Model)

- Test Accuracy: **0.9825**
- Precision: **0.9861**
- Recall: **0.9861**
- F1 Score: **0.9861**

The model correctly classified **112 out of 114 test samples**, with only:
- **1 false positive**
- **1 false negative**

This model showed the **best balance between precision and recall**, which is particularly important in medical diagnosis.
---
### Random Forest Classifier

- Test Accuracy: **0.9561**
- Precision: **0.9589**
- Recall: **0.9722**
- F1 Score: **0.9655**

Random Forest performed well and had strong recall, but it showed signs of **overfitting** due to perfect training accuracy.
---
### Decision Tree Classifier

- Test Accuracy: **0.9123**
- Precision: **0.9559**
- Recall: **0.9028**
- F1 Score: **0.9286**

The Decision Tree had the lowest performance and showed the **most overfitting**, with perfect training accuracy but significantly lower test accuracy.
---
## Conclusion

Logistic Regression was the best-performing model for this dataset. It achieved the highest test accuracy and the best balance between precision and recall. Because the dataset classes are relatively well separated, a linear decision boundary works effectively for classification.
In a medical setting, minimizing both **false negatives (missed cancer cases)** and **false positives (false alarms)** is critical. Logistic Regression provided the best overall reliability for this classification task.
---
## Limitations

Several limitations should be considered:

- The dataset is relatively small (569 samples).
- Only three classification models were tested.
- No hyperparameter tuning was performed.
- The dataset may not fully represent all real-world medical scenarios.
- Overfitting occurred in tree-based models due to limited data.

Despite these limitations, the project demonstrates how machine learning can be used to support medical diagnosis and data-driven decision making.
