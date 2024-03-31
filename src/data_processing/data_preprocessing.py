from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_data(df, output_path='data/processed_data.csv'):
    numeric_columns = df.select_dtypes(include=['number']).columns
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns

    # Impute numeric columns with mean, median, and a constant value (e.g., -1)
    numeric_imputers = {
        'mean': SimpleImputer(strategy='mean'),
        'median': SimpleImputer(strategy='median'),
        'constant': SimpleImputer(strategy='constant', fill_value=-1)
    }
    for strategy, imputer in numeric_imputers.items():
        df[[f'{strategy}_imputed_{col}' for col in numeric_columns]] = imputer.fit_transform(df[numeric_columns])

    # Impute non-numeric columns with most frequent value, constant, and a custom value (e.g., 'unknown')
    non_numeric_imputers = {
        'most_frequent': SimpleImputer(strategy='most_frequent'),
        'constant': SimpleImputer(strategy='constant', fill_value='unknown'),
        'custom_value': SimpleImputer(strategy='constant', fill_value='custom_value')
    }
    for strategy, imputer in non_numeric_imputers.items():
        df[[f'{strategy}_imputed_{col}' for col in non_numeric_columns]] = imputer.fit_transform(
            df[non_numeric_columns].astype(str))

    # Additional preprocessing steps as needed

    # Save the preprocessed data to a CSV file
    df.to_csv(output_path, index=False)

    return df

