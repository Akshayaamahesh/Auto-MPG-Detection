# src/feature_selection/l1_regularization.py
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
import pandas as pd

def l1_regularization(X, y, alpha):
    # Impute missing values
    imputer_numeric = SimpleImputer(strategy='mean')
    imputer_non_numeric = SimpleImputer(strategy='most_frequent')

    numeric_columns = X.select_dtypes(include='number').columns
    non_numeric_columns = X.select_dtypes(exclude='number').columns

    X_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(X[numeric_columns]), columns=numeric_columns)
    X_non_numeric_imputed = pd.DataFrame(imputer_non_numeric.fit_transform(X[non_numeric_columns]), columns=non_numeric_columns)

    # One-hot encoding for non-numeric columns
    X_non_numeric_encoded = pd.get_dummies(X_non_numeric_imputed, columns=non_numeric_columns)

    X_imputed = pd.concat([X_numeric_imputed, X_non_numeric_encoded], axis=1)

    # Initialize Lasso regression model
    model = Lasso(alpha=alpha)
    model.fit(X_imputed, y)

    # Get the coefficients before regularization
    initial_coefficients = model.coef_

    # Apply L1 regularization
    model.coef_ = model.coef_ * (model.coef_ != 0)

    # Get the coefficients after regularization
    coefficients_after_regularization = model.coef_

    # Identify the features that were removed
    removed_features = X_imputed.columns[coefficients_after_regularization == 0]

    # Return coefficients after L1 regularization and removed features
    return coefficients_after_regularization, removed_features
