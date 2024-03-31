# src/feature_selection/tree_based_methods.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import pandas as pd

def tree_based_feature_importance(X, y, n_estimators):
    imputer_numeric = SimpleImputer(strategy='mean')
    imputer_non_numeric = SimpleImputer(strategy='most_frequent')

    numeric_columns = X.select_dtypes(include='number').columns
    non_numeric_columns = X.select_dtypes(exclude='number').columns

    X_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(X[numeric_columns]), columns=numeric_columns)
    X_non_numeric_imputed = pd.DataFrame(imputer_non_numeric.fit_transform(X[non_numeric_columns]), columns=non_numeric_columns)

    # One-hot encoding for non-numeric columns
    X_non_numeric_encoded = pd.get_dummies(X_non_numeric_imputed, columns=non_numeric_columns)

    X_imputed = pd.concat([X_numeric_imputed, X_non_numeric_encoded], axis=1)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_imputed, y)

    # Get feature importances and corresponding feature names
    importances = model.feature_importances_
    feature_names = X_imputed.columns

    # Create a dictionary with feature names and importances
    feature_importance_dict = dict(zip(feature_names, importances))

    # Print feature names and importances
    for feature, importance in feature_importance_dict.items():
        print(f"{feature}: {importance}")

    # Return the feature importances dictionary
    return feature_importance_dict
