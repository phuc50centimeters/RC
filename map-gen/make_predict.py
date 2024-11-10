import joblib
import pandas as pd

model = joblib.load('xgb_model.pkl')

new_data = pd.read_csv('train_data_v1/joined_20201001.csv')
columns_to_drop = ['Label']

predictions = model.predict(new_data.drop(columns=columns_to_drop))

new_data['Predict'] = predictions
new_data.to_csv('predict.csv', index=False)
