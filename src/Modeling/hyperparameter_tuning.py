from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os 
from sklearn.impute import SimpleImputer
import pandas as pd
import optuna
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge, HuberRegressor
from sklearn.linear_model import ElasticNetCV

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

# Define the objective function for Optuna
def objective(trial):
    regressor_name = trial.suggest_categorical('regressor', [
        'Ridge Regression', 'Lasso Regression', 'Decision Tree',
        'Random Forest', 'Gradient Boosting', 'SVR', 'KNN', 'Neural Network',
        'Elastic Net', 'Bayesian Ridge', 'Huber Regressor',
         'Elastic NetCV', 'AdaBoost'
    ])

    if regressor_name == 'Ridge Regression':
        alpha = trial.suggest_loguniform('alpha', 1e-3, 1e3)
        regressor = Ridge(alpha=alpha)
    elif regressor_name == 'Lasso Regression':
        alpha = trial.suggest_loguniform('alpha', 1e-3, 1e3)
        regressor = Lasso(alpha=alpha)
    elif regressor_name == 'Decision Tree':
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        regressor = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
    elif regressor_name == 'Random Forest':
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    elif regressor_name == 'Gradient Boosting':
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        regressor = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    elif regressor_name == 'SVR':
        C = trial.suggest_loguniform('C', 1e-3, 1e3)
        epsilon = trial.suggest_float('epsilon', 0.01, 0.2)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        regressor = SVR(C=C, epsilon=epsilon, kernel=kernel)
    elif regressor_name == 'KNN':
        n_neighbors = trial.suggest_int('n_neighbors', 3, 20)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        regressor = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    elif regressor_name == 'Neural Network':
        hidden_layer_sizes = tuple(
            [trial.suggest_int(f'n_units_l{i}', 5, 100) for i in range(trial.suggest_int('n_layers', 1, 3))]
        )
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
        regressor = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=1000)
    elif regressor_name == 'Elastic Net':
        alpha = trial.suggest_loguniform('alpha', 1e-3, 1e3)
        l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
        regressor = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    elif regressor_name == 'Bayesian Ridge':
        alpha_1 = trial.suggest_loguniform('alpha_1', 1e-6, 1e-4)
        alpha_2 = trial.suggest_loguniform('alpha_2', 1e-6, 1e-4)
        lambda_1 = trial.suggest_loguniform('lambda_1', 1e-6, 1e-4)
        lambda_2 = trial.suggest_loguniform('lambda_2', 1e-6, 1e-4)
        regressor = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2)
    elif regressor_name == 'Huber Regressor':
        epsilon = trial.suggest_float('epsilon', 1.1, 1.5)
        alpha = trial.suggest_loguniform('alpha', 1e-4, 0.01)
        regressor = HuberRegressor(epsilon=epsilon, alpha=alpha)
    elif regressor_name == 'Elastic NetCV':
        l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
        regressor = ElasticNetCV(l1_ratio=l1_ratio, cv=5)
    elif regressor_name == 'AdaBoost':
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0)
        regressor = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate)

    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Optimize hyperparameters for each regressor
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Get the best parameters
best_params = study.best_params
best_regressor_name = best_params['regressor']

# Train the best regressor with the best parameters
best_regressor = None
if best_regressor_name == 'Ridge Regression':
    best_alpha = best_params['alpha']
    best_regressor = Ridge(alpha=best_alpha)
elif best_regressor_name == 'Lasso Regression':
    best_alpha = best_params['alpha']
    best_regressor = Lasso(alpha=best_alpha)
elif best_regressor_name == 'Decision Tree':
    best_max_depth = best_params['max_depth']
    best_min_samples_split = best_params['min_samples_split']
    best_regressor = DecisionTreeRegressor(max_depth=best_max_depth, min_samples_split=best_min_samples_split)
elif best_regressor_name == 'Random Forest':
    best_n_estimators = best_params['n_estimators']
    best_max_depth = best_params['max_depth']
    best_min_samples_split = best_params['min_samples_split']
    best_regressor = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth, min_samples_split=best_min_samples_split)
elif best_regressor_name == 'Gradient Boosting':
    best_n_estimators = best_params['n_estimators']
    best_learning_rate = best_params['learning_rate']
    best_max_depth = best_params['max_depth']
    best_regressor = GradientBoostingRegressor(n_estimators=best_n_estimators, learning_rate=best_learning_rate, max_depth=best_max_depth)
elif best_regressor_name == 'SVR':
    best_C = best_params['C']
    best_epsilon = best_params['epsilon']
    best_kernel = best_params['kernel']
    best_regressor = SVR(C=best_C, epsilon=best_epsilon, kernel=best_kernel)
elif best_regressor_name == 'KNN':
    best_n_neighbors = best_params['n_neighbors']
    best_weights = best_params['weights']
    best_regressor = KNeighborsRegressor(n_neighbors=best_n_neighbors, weights=best_weights)
elif best_regressor_name == 'Neural Network':
    best_hidden_layer_sizes = tuple(best_params[f'n_units_l{i}'] for i in range(best_params['n_layers']))
    best_activation = best_params['activation']
    best_regressor = MLPRegressor(hidden_layer_sizes=best_hidden_layer_sizes, activation=best_activation, max_iter=1000)
elif best_regressor_name == 'Elastic Net':
    best_alpha = best_params['alpha']
    best_l1_ratio = best_params['l1_ratio']
    best_regressor = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)
elif best_regressor_name == 'Bayesian Ridge':
    best_alpha_1 = best_params['alpha_1']
    best_alpha_2 = best_params['alpha_2']
    best_lambda_1 = best_params['lambda_1']
    best_lambda_2 = best_params['lambda_2']
    best_regressor = BayesianRidge(alpha_1=best_alpha_1, alpha_2=best_alpha_2, lambda_1=best_lambda_1, lambda_2=best_lambda_2)
elif best_regressor_name == 'Huber Regressor':
    best_epsilon = best_params['epsilon']
    best_alpha = best_params['alpha']
    best_regressor = HuberRegressor(epsilon=best_epsilon, alpha=best_alpha)
elif best_regressor_name == 'Elastic NetCV':
    best_l1_ratio = best_params['l1_ratio']
    best_regressor = ElasticNetCV(l1_ratio=best_l1_ratio, cv=5)
elif best_regressor_name == 'AdaBoost':
    best_n_estimators = best_params['n_estimators']
    best_learning_rate = best_params['learning_rate']
    best_regressor = AdaBoostRegressor(n_estimators=best_n_estimators, learning_rate=best_learning_rate)

best_regressor.fit(X_train, y_train)

# Evaluate the best regressor on the test set
y_pred = best_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Regressor: {best_regressor_name}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save the best regressor
file_path = os.path.join('trained_models', f'{best_regressor_name}_model.joblib')
joblib.dump(best_regressor, file_path)
