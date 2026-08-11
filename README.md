# MLOps Practical - Machine Learning Pipeline using DVC

This practical demonstrates how to create an end-to-end Machine Learning pipeline using Python, Git, and DVC (Data Version Control).

The pipeline consists of four stages:

```text
Data Collection
       ↓
Data Preprocessing
       ↓
Model Building
       ↓
Model Evaluation



## Step 1: Create the Project Folder

Open the terminal and create a new project folder:

mkdir mlpipeline
cd mlpipeline


## Step 2: Create Project Folders

Create the src and data folders:

mkdir src
mkdir data

The initial project structure will be:

mlpipeline/
│
├── data/
└── src/


## Step 3: Download the Dataset

For this practical, we use the Water Potability Dataset from Kaggle.

Download the dataset from:

https://www.kaggle.com/datasets/adityakadiwal/water-potability

After downloading the dataset, extract it and place the following file inside the data folder:

water_potability.csv

The project structure should now be:

mlpipeline/
│
├── data/
│   └── water_potability.csv
│
└── src/


## Step 4: Create Data Collection File

Inside the src folder, create:

DataCollection.py

Add the following code:

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Read the dataset
data = pd.read_csv(r"data\water_potability.csv")

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


## Step 5: Initialize Git

Initialize the project as a Git repository:

git init


## Step 6: Initialize DVC

Initialize DVC:

dvc init


## Step 7: Create the Data Collection DVC Stage

Create the first DVC pipeline stage:

dvc stage add -n Data_Collection -d src/DataCollection.py -d data/water_potability.csv -o data/raw python src/DataCollection.py

This stage corresponds to:

Data_Collection:
  cmd: python src/DataCollection.py
  deps:
    - src/DataCollection.py
    - data/water_potability.csv
  outs:
    - data/raw


## Step 8: Run the Data Collection Stage

Run:

dvc repro

The Data_Collection stage will execute.

The following files will be generated:

data/
├── water_potability.csv
└── raw/
    ├── train.csv
    └── test.csv


## Step 9: View the DVC Pipeline

Check the pipeline:

dvc dag

At this point, the pipeline contains:

Data_Collection


## Step 10: Create Data Preprocessing File

Create the following file inside the src folder:

DataPreprocessing.py

Add the following code:

import os
import pandas as pd
import numpy as np

# Read the training and testing datasets
train_data = pd.read_csv("./data/raw/train.csv")
test_data = pd.read_csv("./data/raw/test.csv")


def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
    return df


# Fill missing values
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# Create the processed data directory
data_path = os.path.join("data", "processed")
os.makedirs(data_path, exist_ok=True)

# Save the processed datasets
train_processed_data.to_csv(
    os.path.join(data_path, "train_processed.csv"),
    index=False
)

test_processed_data.to_csv(
    os.path.join(data_path, "test_processed.csv"),
    index=False
)

print("Data preprocessing completed successfully!")


## Step 11: Create the Data Preprocessing DVC Stage

Create the second DVC stage:

dvc stage add -n Data_Preprocessing -d src/DataPreprocessing.py -d data/raw -o data/processed python src/DataPreprocessing.py

This stage corresponds to:

Data_Preprocessing:
  cmd: python src/DataPreprocessing.py
  deps:
    - src/DataPreprocessing.py
    - data/raw
  outs:
    - data/processed


## Step 12: Reproduce the Pipeline

Run:

dvc repro

The pipeline will execute:

Data_Collection
       ↓
Data_Preprocessing

The processed files will be generated:

data/
└── processed/
    ├── train_processed.csv
    └── test_processed.csv


## Step 13: Check the DVC DAG

Run:

dvc dag

The pipeline should now look like:

Data_Collection
       ↓
Data_Preprocessing


## Step 14: Create Model Building File

Create the following file:

src/ModelBuilding.py

Add the following code:

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

# Read the training dataset
train_data = pd.read_csv("./data/processed/train_processed.csv")

# Split features and target
X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values

# Train the model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save the trained model
pickle.dump(clf, open("data/model.pkl", "wb"))


## Step 15: Create the Model Building DVC Stage

Create the third DVC stage:

dvc stage add -n Model_Building -d src/ModelBuilding.py -d data/processed -o data/model.pkl python src/ModelBuilding.py

This stage corresponds to:

Model_Building:
  cmd: python src/ModelBuilding.py
  deps:
    - src/ModelBuilding.py
    - data/processed
  outs:
    - data/model.pkl


## Step 16: Reproduce the Pipeline

Run:

dvc repro

The pipeline will execute:

Data_Collection
       ↓
Data_Preprocessing
       ↓
Model_Building

The trained model will be generated:

data/model.pkl


## Step 17: Check the DVC DAG

Run:

dvc dag

The pipeline should now look like:

Data_Collection
       ↓
Data_Preprocessing
       ↓
Model_Building


## Step 18: Create Model Evaluation File

Create:

src/Metrics.py

Add the following code:

import pandas as pd
import pickle
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load test dataset
test_data = pd.read_csv("./data/processed/test_processed.csv")

# Split features and target
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Load trained model
model = pickle.load(open("data/model.pkl", "rb"))

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

# Store metrics in a dictionary
metrics_dict = {
    "acc": acc,
    "precision": pre,
    "recall": recall,
    "f1_score": f1score
}

# Save metrics in JSON format
with open("data/metrics.json", "w") as file:
    json.dump(metrics_dict, file, indent=4)


## Step 19: Create the Model Evaluation DVC Stage

Create the fourth DVC stage:

dvc stage add -n Model_Evaluation -d src/Metrics.py -d data/model.pkl -M data/metrics.json python src/Metrics.py

This stage corresponds to:

Model_Evaluation:
  cmd: python src/Metrics.py
  deps:
    - src/Metrics.py
    - data/model.pkl
  metrics:
    - data/metrics.json:
        cache: false


## Step 20: Run the Complete Pipeline

Now run the complete DVC pipeline:

dvc repro

The complete pipeline will execute in the following order:

Data_Collection
       ↓
Data_Preprocessing
       ↓
Model_Building
       ↓
Model_Evaluation


## Step 21: View the Complete DVC DAG

Run:

dvc dag

The final pipeline will look similar to:

+-------------------+
|  Data_Collection  |
+---------+---------+
          |
          v
+-----------------------+
| Data_Preprocessing    |
+----------+------------+
           |
           v
+-------------------+
|  Model_Building   |
+---------+---------+
          |
          v
+---------------------+
|  Model_Evaluation   |
+---------------------+


## Step 22: Check DVC Status

Check the status of the pipeline:

dvc status

This shows whether any dependencies or outputs have changed.


## Step 23: Display Model Metrics

The evaluation stage generates:

data/metrics.json

Display the metrics using:

dvc metrics show

The metrics include:

- Accuracy
- Precision
- Recall
- F1 Score


## Step 24: View the Metrics JSON File

On Windows:

type data\metrics.json

On Linux/macOS:

cat data/metrics.json


## Step 25: Check the Final Project Structure

After completing all stages, the project structure will look like:

mlpipeline/
│
├── .dvc/
│
├── data/
│   ├── water_potability.csv
│   │
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   ├── processed/
│   │   ├── train_processed.csv
│   │   └── test_processed.csv
│   │
│   ├── model.pkl
│   └── metrics.json
│
├── src/
│   ├── DataCollection.py
│   ├── DataPreprocessing.py
│   ├── ModelBuilding.py
│   └── Metrics.py
│
├── dvc.yaml
├── dvc.lock
└── README.md


## Step 26: Check Git Status

Check the Git status:

git status


## Step 27: Add Files to Git

Add the project files:

git add .


## Step 28: Commit the Project

Commit the project:

git commit -m "Create MLOps pipeline using DVC"


## Step 29: Connect the Local Repository to GitHub

Create a repository on GitHub named:

MLOps

Then connect the local repository to GitHub:

git remote add origin https://github.com/OV2294/MLOps.git

Check the remote:

git remote -v


## Step 30: Push the Project to GitHub

Rename the branch to main:

git branch -M main

Push the project:

git push -u origin main


# Complete DVC Stage Commands

## Data Collection

dvc stage add -n Data_Collection -d src/DataCollection.py -d data/water_potability.csv -o data/raw python src/DataCollection.py

## Data Preprocessing

dvc stage add -n Data_Preprocessing -d src/DataPreprocessing.py -d data/raw -o data/processed python src/DataPreprocessing.py

## Model Building

dvc stage add -n Model_Building -d src/ModelBuilding.py -d data/processed -o data/model.pkl python src/ModelBuilding.py

## Model Evaluation

dvc stage add -n Model_Evaluation -d src/Metrics.py -d data/model.pkl -M data/metrics.json python src/Metrics.py


# Important DVC Commands

dvc init
Initialize DVC.

dvc stage add
Create a DVC pipeline stage.

dvc repro
Run or reproduce the pipeline.

dvc dag
Display the pipeline graph.

dvc status
Check pipeline status.

dvc metrics show
Display model metrics.

dvc metrics diff
Compare metrics.


# Final Pipeline

                  MLOps Pipeline
                       |
                       v
              Data Collection
                       |
                       v
             Data Preprocessing
                       |
                       v
               Model Building
                       |
                       v
              Model Evaluation
                       |
                       v
                 metrics.json


# Conclusion

This practical demonstrates how to build a reproducible Machine Learning pipeline using Python, Git, and DVC.

The pipeline performs the following operations:

1. Download the Water Potability dataset.
2. Create the Data Collection stage.
3. Split the dataset into training and testing data.
4. Create the Data Preprocessing stage.
5. Handle missing values using the median.
6. Create the Model Building stage.
7. Train a Random Forest Classifier.
8. Create the Model Evaluation stage.
9. Calculate Accuracy, Precision, Recall, and F1 Score.
10. Store the evaluation metrics in metrics.json.
11. Reproduce the complete pipeline using dvc repro.
12. Visualize the pipeline using dvc dag.
13. Check pipeline status using dvc status.
14. Display model metrics using dvc metrics show.
15. Track the project using Git.
16. Commit the project.
17. Push the project to GitHub.
