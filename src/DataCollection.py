import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Read the dataset
data = pd.read_csv(r"C:\Users\omkar\Downloads\water_potability.csv")

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42
)

# Create the output directory
data_path = os.path.join("data", "raw")
os.makedirs(data_path, exist_ok=True)

# Save the train and test datasets
train_data.to_csv(
    os.path.join(data_path, "train.csv"),
    index=False
)

test_data.to_csv(
    os.path.join(data_path, "test.csv"),
    index=False
)

print("Train and Test datasets saved successfully!")