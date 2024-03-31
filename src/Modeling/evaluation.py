# evaluation.py
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os 
from sklearn.impute import SimpleImputer
import pandas as pd

# Load the MPG dataset
file_path = 'Data\processed_data.csv' 
df = pd.read_csv(file_path)

# Assuming 'mpg' is the target variable
X = df.drop('mpg', axis=1)
y = df['mpg']

imputer_numeric = SimpleImputer(strategy='mean')
imputer_non_numeric = SimpleImputer(strategy='most_frequent')

numeric_columns = X.select_dtypes(include='number').columns
non_numeric_columns = X.select_dtypes(exclude='number').columns

X_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(X[numeric_columns]), columns=numeric_columns)
X_non_numeric_imputed = pd.DataFrame(imputer_non_numeric.fit_transform(X[non_numeric_columns]), columns=non_numeric_columns)

    # One-hot encoding for non-numeric columns
X_non_numeric_encoded = pd.get_dummies(X_non_numeric_imputed, columns=non_numeric_columns)

X_imputed = pd.concat([X_numeric_imputed, X_non_numeric_encoded], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Load trained regressors
trained_regressors = {}
regressor_names = [
    'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Decision Tree',
    'Random Forest', 'Gradient Boosting', 'SVR', 'KNN', 'Neural Network',
    'Elastic Net', 'Bayesian Ridge', 'Huber Regressor',
    'Gaussian Process', 'Elastic NetCV', 'AdaBoost'
]

for name in regressor_names:
    file_path1 = os.path.join('trained_models', f'{name}_model.joblib')
    trained_regressors[name] = joblib.load(file_path)

# Evaluate each regressor
for name, regressor in trained_regressors.items():
    y_pred = regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Regressor: {name}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print("\n")
