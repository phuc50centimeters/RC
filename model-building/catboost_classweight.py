import pandas as pd
from catboost import CatBoostClassifier
from joblib import dump, load
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Load your dataset
df = pd.read_csv("../data-statistic/combined_data.csv")
X = df.drop(columns=["Label", "row", "col", "month", "day", "hour"])
y = df["Label"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define the CatBoost model
best_params = {
    'iterations': 319,
    'learning_rate': 0.253,
    'depth': 9,
    'colsample_bylevel': 0.8554,
    'l2_leaf_reg': 0.809,
    'border_count': 74,
    'subsample': 0.933,
    'random_state': 42,
    'verbose': 3,
    'auto_class_weights': "Balanced"
}

model = CatBoostClassifier(**best_params)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report - Model with Best Parameters:")
print(classification_report(y_test, y_pred))

dump(model, "catboost_best_model_trained.joblib")
