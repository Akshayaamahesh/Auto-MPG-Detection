# src/feature_selection/recursive_feature_elimination.py
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
import pandas as pd

def rfe_feature_selection(X, y, n_features_to_select):
    imputer_numeric = SimpleImputer(strategy='mean')
    imputer_non_numeric = SimpleImputer(strategy='most_frequent')

    numeric_columns = X.select_dtypes(include='number').columns
    non_numeric_columns = X.select_dtypes(exclude='number').columns

    X_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(X[numeric_columns]), columns=numeric_columns)
    X_non_numeric_imputed = pd.DataFrame(imputer_non_numeric.fit_transform(X[non_numeric_columns]), columns=non_numeric_columns)

    # One-hot encoding for non-numeric columns
    X_non_numeric_encoded = pd.get_dummies(X_non_numeric_imputed, columns=non_numeric_columns)

    X_imputed = pd.concat([X_numeric_imputed, X_non_numeric_encoded], axis=1)

    # Initialize RFE with a Ridge regression model
    estimator = Ridge()  # You can adjust other parameters if needed

    # Use RFE for feature selection
    rfe = RFE(estimator, n_features_to_select=n_features_to_select)
    X_rfe = rfe.fit_transform(X_imputed, y)

    # Get the selected feature names
    selected_feature_names = X_imputed.columns[rfe.support_]

    # Print the selected feature names
    print("Selected features:", selected_feature_names)

    return X_rfe