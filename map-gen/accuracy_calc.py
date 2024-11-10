import pandas as pd

data = pd.read_csv('predict.csv')

accuracy = (data['Label'] == data['Predict']).mean()
print(f'Accuracy: {accuracy:.2%}')

