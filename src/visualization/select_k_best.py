# src/feature_selection/select_k_best.py
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
import pandas as pd

def select_k_best_features(X, y, k):
    imputer_numeric = SimpleImputer(strategy='mean')
    imputer_non_numeric = SimpleImputer(strategy='most_frequent')

    numeric_columns = X.select_dtypes(include='number').columns
    non_numeric_columns = X.select_dtypes(exclude='number').columns

    X_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(X[numeric_columns]), columns=numeric_columns)
    X_non_numeric_imputed = pd.DataFrame(imputer_non_numeric.fit_transform(X[non_numeric_columns]), columns=non_numeric_columns)

    # One-hot encoding for non-numeric columns
    X_non_numeric_encoded = pd.get_dummies(X_non_numeric_imputed, columns=non_numeric_columns)

    X_imputed = pd.concat([X_numeric_imputed, X_non_numeric_encoded], axis=1)
    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(X_imputed, y)

    # Get selected feature indices
    selected_feature_indices = selector.get_support(indices=True)

    # Get selected feature names
    selected_feature_names = X_imputed.columns[selected_feature_indices]

    # Display selected features
    print("Selected Features:", selected_feature_names)

    # Return selected features
    return X_new

