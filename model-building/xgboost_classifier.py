import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from joblib import dump, load


df = pd.read_csv('../data-statistic/combined_data.csv')
X = df.drop(columns=['Label'])
y = df['Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize XGBoost classifier
#xgb_model = XGBClassifier(
#    
#    random_state=42)
xgb_model = XGBClassifier(
    colsample_bytree=0.9961057796456088,
    gamma=0.30874075481385826,
    learning_rate=0.19349594814648427,
    max_depth=9,
    reg_alpha=0.023062425041415757,
    reg_lambda=0.5247746602583891,
    subsample=0.6999304858576277,
    n_estimators=385,
    eval_metric="logloss",   
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model
dump(xgb_model, 'xgb_model.joblib')

