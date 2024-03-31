# src/feature_selection/variance_threshold.py
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
import pandas as pd

def variance_threshold_selection(X, threshold):
    imputer_numeric = SimpleImputer(strategy='mean')
    imputer_non_numeric = SimpleImputer(strategy='most_frequent')

    numeric_columns = X.select_dtypes(include='number').columns
    non_numeric_columns = X.select_dtypes(exclude='number').columns

    X_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(X[numeric_columns]), columns=numeric_columns)
    X_non_numeric_imputed = pd.DataFrame(imputer_non_numeric.fit_transform(X[non_numeric_columns]), columns=non_numeric_columns)

    # One-hot encoding for non-numeric columns
    X_non_numeric_encoded = pd.get_dummies(X_non_numeric_imputed, columns=non_numeric_columns)

    X_imputed = pd.concat([X_numeric_imputed, X_non_numeric_encoded], axis=1)

    selector = VarianceThreshold(threshold=threshold)
    X_high_variance = selector.fit_transform(X_imputed)
    # Return features with high variance
    return X_high_variance
