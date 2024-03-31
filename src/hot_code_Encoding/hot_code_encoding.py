import pandas as pd
import matplotlib.pyplot as plt
import os

# Step 1: Load the Dataset
mpg_data = pd.read_csv('C:/Users/aksha/Crisp DM/Data/processed_data.csv')

# Step 2: Identify Discrete and Continuous Variables
discrete_vars = ['cylinders', 'origin']
continuous_vars = ['mpg']  # Add other continuous variables as needed

# Get the current working directory
current_directory = os.getcwd()

# Create the 'plots' directory within the current working directory
save_directory = os.path.join(current_directory, 'src','hot_code_Encoding','plots')

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Display summary statistics for numerical columns
print(mpg_data.describe())

# Step 3: Analyze Discrete Variables and Save Plots
for var in discrete_vars:
    plt.figure(figsize=(8, 6))
    plt.bar(mpg_data[var].value_counts().index, mpg_data[var].value_counts(), edgecolor='black')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {var} in the MPG Dataset')
    plt.savefig(os.path.join(save_directory, f'{var}_distribution.png'))
    plt.close()

# Step 4: One-Hot Encoding for Discrete Variables
for var in discrete_vars:
    one_hot_var = pd.get_dummies(mpg_data[var], prefix=var)
    mpg_data = pd.concat([mpg_data, one_hot_var], axis=1)

# Step 5: Continuously Variables Analysis and Save Plots
for var in continuous_vars:
    plt.figure(figsize=(8, 6))
    plt.hist(mpg_data[var], bins=20, edgecolor='black')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {var} in the MPG Dataset')
    plt.savefig(os.path.join(save_directory, f'{var}_distribution.png'))
    plt.close()
