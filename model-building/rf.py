import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump, load
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('../data-statistic/combined_data.csv')
X = df.drop(columns=['Label',])
y = df['Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


rf = RandomForestClassifier(
    n_estimators=550,
    max_depth=50,
    min_samples_split=11,
    min_samples_leaf=5,
    max_features=None,
    class_weight="balanced_subsample",
    random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

dump(rf, 'rf.joblib')

