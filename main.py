from src.data_processing.data_loader import load_data
from src.data_processing.data_preprocessing import preprocess_data

# Load raw data
raw_data_path = 'Data/mpg.csv'
raw_data = load_data(raw_data_path)

# Preprocess the data
processed_data = preprocess_data(raw_data)