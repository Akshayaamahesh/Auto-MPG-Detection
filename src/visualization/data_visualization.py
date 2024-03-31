import seaborn as sns
import matplotlib.pyplot as plt

def visualize_data(df):
    # Assuming 'mpg' is the target variable
    target_variable = 'mpg'

    # Create a pair plot
    sns.pairplot(df, hue=target_variable)
    plt.title(f'Pair Plot of Features with {target_variable}')
    plt.savefig('data_visualization.png')
    plt.show()
