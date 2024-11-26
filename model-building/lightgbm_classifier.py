import pandas as pd
import lightgbm as lgb
from joblib import dump, load
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform


# Load your dataset
df = pd.read_csv("../data-statistic/combined_data.csv")
X = df.drop(columns=["Label", "row", "col", "month", "day", "hour"])
y = df["Label"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_params = {
    'n_estimators': 383,
    'learning_rate': 0.135,
    'num_leaves': 63,
    'max_depth': 9, 
    'colsample_bytree': 0.7553,   
    'min_child_samples': 60,     
    'subsample': 0.622,         
    'auto_class_weights': "Balanced",
    'random_state': 42,           
    'objective': 'binary',       
    'metric': 'binary_error'    
}

lgbm_model = lgb.LGBMClassifier(**best_params)

lgbm_model.fit(X_train, y_train)

y_pred_best = lgbm_model.predict(X_test)

print("Classification Report - Model with Best Parameters:")
print(classification_report(y_test, y_pred_best))

dump(lgbm_model, "best_lightgbm_model_trained.joblib")

