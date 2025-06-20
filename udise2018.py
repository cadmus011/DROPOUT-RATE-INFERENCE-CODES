import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
file_path = 'eesaahmed197_17484137453452513.csv'
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found in directory '{os.getcwd()}'.\nAvailable files: {os.listdir()}")
    exit()

df = pd.read_csv(file_path)
df['Year'] = df['Year'].str.extract(r'(\d{4})').astype(int)
target_col = "Drop-Out Rate Of Secondary School (UOM:Ratio), Scaling Factor:1"
df = df[~df[target_col].isna()]

train_data = df[df['Year'].between(2014, 2017)]
features = ['State', 'Gender', 'Year', "Drop-Out Rate Of Secondary School (UOM:Ratio), Scaling Factor:1"]
model = Pipeline([
    ('preprocess', ColumnTransformer([
        ('encode', OneHotEncoder(handle_unknown='ignore'), ['State', 'Gender'])
    ], remainder='passthrough')),
    ('regressor', RandomForestRegressor(n_estimators=150, random_state=42))
])

model.fit(train_data[features], train_data[target_col])

latest_data = df[df['Year'] == 2017].copy()
predict_2018 = latest_data[features].copy()
predict_2018['Year'] = 2018 


predictions = model.predict(predict_2018)
predict_2018['Predicted_2018_Dropout'] = np.round(predictions, 2)

pd.set_option('display.max_rows', None)
print("\n2018 Secondary School Dropout Rate Predictions (all rows):")
print(predict_2018[['State', 'Gender', 'Year', 'Predicted_2018_Dropout']].to_string(index=False))
pd.reset_option('display.max_rows')

print(f"\nNumber of predictions: {len(predict_2018)}")
print(f"Mean predicted dropout rate: {predict_2018['Predicted_2018_Dropout'].mean():.2f}")
print(f"Max predicted dropout rate: {predict_2018['Predicted_2018_Dropout'].max():.2f}")
print(f"Min predicted dropout rate: {predict_2018['Predicted_2018_Dropout'].min():.2f}")

predict_2018[['State', 'Gender', 'Year', 'Predicted_2018_Dropout']].to_csv('2018_dropout_predictions.csv', index=False)
print("\nPredictions saved to '2018_dropout_predictions.csv'")
