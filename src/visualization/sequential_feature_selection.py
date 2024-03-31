# src/feature_selection/sequential_feature_selection.py
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
import pandas as pd


def sequential_feature_selection(X, y, k_features):
    imputer_numeric = SimpleImputer(strategy='mean')
    imputer_non_numeric = SimpleImputer(strategy='most_frequent')

    numeric_columns = X.select_dtypes(include='number').columns
    non_numeric_columns = X.select_dtypes(exclude='number').columns

    X_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(X[numeric_columns]), columns=numeric_columns)
    X_non_numeric_imputed = pd.DataFrame(imputer_non_numeric.fit_transform(X[non_numeric_columns]), columns=non_numeric_columns)

    # One-hot encoding for non-numeric columns
    X_non_numeric_encoded = pd.get_dummies(X_non_numeric_imputed, columns=non_numeric_columns)

    X_imputed = pd.concat([X_numeric_imputed, X_non_numeric_encoded], axis=1)
    model = Ridge()
    sfs = SequentialFeatureSelector(model, k_features=k_features, forward=True, scoring='r2', cv=5)
    sfs.fit(X_imputed, y)
    # Return selected features
    return sfs.transform(X)
