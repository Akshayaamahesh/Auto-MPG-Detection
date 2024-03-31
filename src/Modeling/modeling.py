'''from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
#from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor
#from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize regressors
regressors = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    #'XGBoost': XGBRegressor(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'Neural Network': MLPRegressor(),
    'Elastic Net': ElasticNet(),
    'Bayesian Ridge': BayesianRidge(),
    'Huber Regressor': HuberRegressor(),
    'Isotonic Regression': IsotonicRegression(out_of_bounds='clip'),
    'Gaussian Process': GaussianProcessRegressor(),
    #'CatBoost': CatBoostRegressor(),
    #'LightGBM': LGBMRegressor(),
    'Elastic NetCV': ElasticNetCV(),
    #'LGBM Regressor': LGBMRegressor(),
    'AdaBoost': AdaBoostRegressor()
}

def evaluate_regressor(regressor, X_train, X_test, y_train, y_test):
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Regressor: {type(regressor).__name__}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print("\n")

# Train and evaluate each regressor
for _, regressor in regressors.items():
    evaluate_regressor(regressor, X_train, X_test, y_train, y_test)
'''
# modeling.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.gaussian_process import GaussianProcessRegressor
import joblib
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



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize regressors
regressors = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'Neural Network': MLPRegressor(),
    'Elastic Net': ElasticNet(),
    'Bayesian Ridge': BayesianRidge(),
    'Huber Regressor': HuberRegressor(),
    'Gaussian Process': GaussianProcessRegressor(),
    'Elastic NetCV': ElasticNetCV(),
    'AdaBoost': AdaBoostRegressor()
}

# Train each regressor
trained_regressors = {}
for name, regressor in regressors.items():
    regressor.fit(X_train, y_train)
    trained_regressors[name] = regressor

# Save trained regressors
for name, regressor in trained_regressors.items():
    joblib.dump(regressor, f'{name}_model.joblib')
