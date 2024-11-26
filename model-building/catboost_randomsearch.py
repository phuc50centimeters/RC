from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import uniform, randint
import joblib
import pandas as pd

# Load your dataset
df = pd.read_csv('../data-statistic/combined_data.csv')
X = df.drop(columns=['Label', 'year'])
y = df['Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate scale_pos_weight
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# Initialize CatBoost classifier
catboost_model = CatBoostClassifier(
    eval_metric='Logloss',
    random_state=42,
    class_weights=[scale_pos_weight, 1],
    verbose=0 
)

# Define the parameter grid for RandomizedSearchCV
param_distributions = {
    'iterations': randint(150, 400),     
    'depth': randint(5, 10),            
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.5, 0.5),    
    'colsample_bylevel': uniform(0.5, 0.5),
    'l2_leaf_reg': uniform(0, 1), 
    'border_count': randint(32, 255),
}

# Initialize RandomizedSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=catboost_model,
    param_distributions=param_distributions,
    n_iter=50,  
    scoring='f1',
    cv=cv,      
    verbose=3, 
    random_state=42,
    n_jobs=-1 
)

# Fit the model
random_search.fit(X_train, y_train)

# Print best parameters and best score
print("Best parameters:", random_search.best_params_)
print("Best cross-validation F1 score:", random_search.best_score_)

# Evaluate the model with the best parameters on the test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the best model
joblib.dump(best_model, 'catboost_model_randomsearch.joblib')

