import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import os

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import random



# Load your dataset
# Load the dataset
file_path = 'D:\\1 code AI'
file_names = ['new_data_v1.csv']
datasets = [pd.read_csv(os.path.join(file_path, file)) for file in file_names]
dataset = pd.concat(datasets, ignore_index=True)

# Count the number of labels 0 and 1
label_counts = dataset['Label'].value_counts()
print("Label counts:")
print(label_counts)

dataset.info()

from sklearn.model_selection import train_test_split

# Check if columns exist before dropping
columns_drop = ["Label"]
existing_columns_to_drop = [col for col in columns_drop if col in dataset.columns]

X = dataset.drop(columns=existing_columns_to_drop)
y = dataset["Label"]

print(X.head())
print(X.info())
print(y.head())
print(y.value_counts())


# Split the data into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

print("Training set label distribution:")
print(y_train.value_counts())
print("Test set label distribution:")
print(y_test.value_counts())


#random search
param_dist = {
    "n_estimators": np.arange(100, 1000, 100),  
    'max_features': [None, 'sqrt', 'log2'],     
    'max_depth': np.arange(10, 110, 10),        
    'min_samples_split': np.arange(2, 23, 3),   
    'min_samples_leaf': np.arange(1, 22, 3),    
    'class_weight': ['balanced', 'balanced_subsample']  
}




rf = RandomForestClassifier(random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_random_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                       n_iter=50, cv=skf, scoring='f1', n_jobs=-1, verbose=10, random_state=42)

rf_random_search.fit(X_train, y_train)

print("Best parameters found: ", rf_random_search.best_params_)

best_rf = rf_random_search.best_estimator_

y_pred = best_rf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

