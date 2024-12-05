# Loan Prediction Project

## Overview

This project aims to predict loan eligibility using machine learning models. We utilize three different algorithms: Logistic Regression, Decision Tree Classifier, and Support Vector Machine (SVM), and evaluate their performance.

## Dataset

The project uses the 'Loan prediction train.csv' dataset. It contains information about loan applicants, such as gender, marital status, education, income, loan amount, and loan status (approved or rejected).

## Data Preprocessing

The following data preprocessing steps were performed:

1. **Handling Missing Values:** Missing values were imputed using appropriate strategies, such as replacing with the most frequent value or using the mean. For example, missing values in the 'Gender' column were filled with 'Male', and missing values in the 'Loan_Amount_Term' column were filled with 12.0.
2. **Data Transformation:** Categorical features were converted into numerical representations using label encoding. For example, 'Gender' was encoded as 0 for 'Male' and 1 for 'Female'.
3. **Data Splitting:** The dataset was split into training and testing sets using `train_test_split` with a `test_size` of 0.2 and a `random_state` of 5. This ensures that 80% of the data is used for training and 20% for testing.

## Models

The following machine learning models were trained and evaluated:

1. **Logistic Regression:** A linear model used for binary classification.
2. **Decision Tree Classifier:** A tree-based model that makes decisions based on a series of conditions.
3. **Support Vector Machine (SVM):** A powerful model that finds the optimal hyperplane to separate data points into different classes.

## Evaluation

Model performance was evaluated using the following metrics:

* **Classification Report:** Provides precision, recall, F1-score, and support for each class.
* **Confusion Matrix:** Shows the number of true positives, true negatives, false positives, and false negatives.

## Results

The results of the models are presented in the notebook. You can find the classification reports and confusion matrices for each model. You can compare the performance of the models based on these metrics to determine which model performs best for this task.

## Conclusion

This project demonstrates the application of machine learning for loan prediction. The results show the effectiveness of Logistic Regression, Decision Tree Classifier, and SVM models in predicting loan eligibility. Further improvements can be explored by trying different hyperparameters or feature engineering techniques.

## Usage

1. Upload the 'Loan prediction train.csv' dataset to your Google Colab environment.
2. Run the code in the notebook to train and evaluate the models.
3. Review the results and compare the performance of the models.

## Dependencies

The project requires the following libraries:

* pandas
* scikit-learn

You can install them using `pip install pandas scikit-learn`.
