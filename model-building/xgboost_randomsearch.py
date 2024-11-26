import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score, RandomizedSearchCV
from xgboost import XGBClassifier
from joblib import dump, load
from scipy.stats import uniform, randint
import joblib

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Load your dataset
df = pd.read_csv('../data-statistic/combined_data.csv')
X = df.drop(columns=['Label', 'year', 'month', 'day', 'hour'])
y = df['Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate scale_pos_weight
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# Initialize XGBoost classifier
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight)

# Define the parameter grid
param_distributions = {
    'n_estimators': randint(150, 400),
    'max_depth': randint(5, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'gamma': uniform(0, 0.5),            
    'reg_alpha': uniform(0, 1),         
    'reg_lambda': uniform(0, 1)        
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
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
print("Best cross-validation accuracy:", random_search.best_score_)

# Evaluate the model with the best parameters on the test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(best_model, 'xgboost_model_randomsearch.joblib')

