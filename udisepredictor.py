import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
file_path = 'eesaahmed197_17484137453452513.csv'
df = pd.read_csv(file_path)

df['YearNum'] = df['Year'].str.extract(r'(\d{4})').astype(int)

target_col = "Drop-Out Rate Of Secondary School (UOM:Ratio), Scaling Factor:1"
df = df[~df[target_col].isna()]

train = df[df['YearNum'].isin([2014, 2015, 2016])].copy()
test = df[df['YearNum'] == 2017].copy()

feature_cols = [
    'State', 'Gender', 'YearNum',
    "Drop-Out Rate Of Primary School (UOM:Ratio), Scaling Factor:1",
    "Drop-Out Rate Of Upper Primary School (UOM:Ratio), Scaling Factor:1",
    "Drop-Out Rate Of Elementary School (UOM:Ratio), Scaling Factor:1"
]

X_train = train[feature_cols]
y_train = train[target_col]
X_test = test[feature_cols]
y_test = test[target_col]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['State', 'Gender'])
    ],
    remainder='passthrough'
)

# Model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
results = test[['State', 'Gender', 'Year', target_col]].copy()
results['Predicted'] = y_pred
results['Error'] = results[target_col] - results['Predicted']

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")

print("\nSample predictions vs. actual (top 10 by error):")
print(results[['State', 'Gender', 'Year', target_col, 'Predicted', 'Error']].sort_values('Error', key=np.abs, ascending=False).head(10))

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Dropout Rate (2017)")
plt.ylabel("Predicted Dropout Rate (2017)")
plt.title("Actual vs Predicted Secondary Dropout Rate (2017)")
plt.grid(True)
plt.tight_layout()
plt.show()

results.to_csv('secondary_dropout_2017_predictions_vs_actual.csv', index=False)
