import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score, RandomizedSearchCV
from xgboost import XGBClassifier
from joblib import dump, load
from scipy.stats import uniform, randint

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Load your dataset
df = pd.read_csv('../data-statistic/new_data_v1.csv')
X = df.drop(columns=['Label'])
y = df['Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate scale_pos_weight
scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]

# Initialize XGBoost classifier
xgb_model = XGBClassifier(eval_metric='auc', random_state=42, scale_pos_weight=scale_pos_weight)

# Define the parameter grid
param_distributions = {
    'n_estimators': randint(150, 400),         # Số lượng cây
    'max_depth': randint(5, 10),              # Độ sâu tối đa của mỗi cây
    'learning_rate': uniform(0.01, 0.3),      # Tốc độ học
    'subsample': uniform(0.5, 0.5),           # Tỷ lệ mẫu cho mỗi cây
    'colsample_bytree': uniform(0.5, 0.5),    # Tỷ lệ đặc trưng cho mỗi cây
    'gamma': uniform(0, 0.5),                 # Giá trị gamma
    'reg_alpha': uniform(0, 1),               # Regularization L1
    'reg_lambda': uniform(0, 1)               # Regularization L2
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=10,                    # Số lần lặp ngẫu nhiên để thử
    scoring='f1',           # Thước đo đánh giá
    cv=cv,                         # Số lượng K-folds trong cross-validation
    verbose=3,
    random_state=42,
    n_jobs=-1                     # Sử dụng tất cả các lõi CPU
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

joblib.dump(best_model, 'xgboost_model.joblib')

