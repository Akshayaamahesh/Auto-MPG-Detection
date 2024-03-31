# src/feature_selection/mutual_information.py
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
import pandas as pd

def mutual_information_selection(X, y):
    imputer_numeric = SimpleImputer(strategy='mean')
    imputer_non_numeric = SimpleImputer(strategy='most_frequent')

    numeric_columns = X.select_dtypes(include='number').columns
    non_numeric_columns = X.select_dtypes(exclude='number').columns

    X_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(X[numeric_columns]), columns=numeric_columns)
    X_non_numeric_imputed = pd.DataFrame(imputer_non_numeric.fit_transform(X[non_numeric_columns]), columns=non_numeric_columns)

    # One-hot encoding for non-numeric columns
    X_non_numeric_encoded = pd.get_dummies(X_non_numeric_imputed, columns=non_numeric_columns)

    X_imputed = pd.concat([X_numeric_imputed, X_non_numeric_encoded], axis=1)

    mi_scores = mutual_info_regression(X_imputed, y)
    # Return mutual information scores for each feature
    return mi_scores
