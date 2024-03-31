from data_visualization import visualize_data
import pandas as pd
from recursive_feature_elimination import rfe_feature_selection
import numpy as np
from sklearn.impute import SimpleImputer
from l1_regularization import l1_regularization
from mutual_information import mutual_information_selection
from tree_based_methods import tree_based_feature_importance
from select_k_best import select_k_best_features
#from sequential_feature_selection import sequential_feature_selection
from variance_threshold import variance_threshold_selection


def feature_sel():
    file_path = 'C:/Users/aksha/Crisp DM/Data/processed_data.csv'  
    df = pd.read_csv(file_path) 
    df.replace('?', np.nan, inplace=True)
    df['mpg'] = pd.to_numeric(df['mpg'], errors='coerce')

    visualize_data(df)
    
    # Example usage of RFE
    X = df.drop('mpg', axis=1)  # Adjust the target variable name
    y = df['mpg']
    X_rfe = rfe_feature_selection(X, y, n_features_to_select=5)
    print("Selected features using RFE:", X_rfe.shape[1])
    
    # Example usage of L1 Regularization
    alpha = 0.01  # Adjust alpha value
    l1_coefficients, removed_features = l1_regularization(X, y, alpha)
    print("Coefficients after L1 Regularization:", l1_coefficients)
    print("Removed Features:", removed_features)

    # Example usage of Tree-based Feature Importance
    n_estimators = 100  # Adjust the number of estimators
    tree_importances = tree_based_feature_importance(X, y, n_estimators)
    print("Feature Importances from Random Forest:")
    for feature, importance in tree_importances.items():
      print(f"{feature}: {importance}")

    # Example usage of Variance Threshold
    threshold = 0.01  # Adjust the threshold value
    X_high_variance = variance_threshold_selection(X, threshold)
    print("Selected features using Variance Threshold:", X_high_variance.shape[1])

    # Example usage of Mutual Information
    mi_scores = mutual_information_selection(X, y)
    print("Mutual Information Scores for each feature:", mi_scores)

    # Example usage of SelectKBest
    X_k_best = select_k_best_features(X, y, k=5)
    print("Selected features using SelectKBest:", X_k_best.shape[1])
'''
    # Example usage of Sequential Feature Selection
    k_features = 5  # Adjust the number of features to select
    X_sfs = sequential_feature_selection(X, y, k_features)
    print("Selected features using Sequential Feature Selection:", X_sfs.shape[1])
'''
if __name__ == "__main__":
    feature_sel()

