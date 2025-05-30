import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
file_path = 'eesaahmed197_17484137453452513.csv'
df = pd.read_csv(file_path)

# Preprocess year
df['YearNum'] = df['Year'].str.extract(r'(\d{4})').astype(int)

# Target column
target_col = "Drop-Out Rate Of Secondary School (UOM:Ratio), Scaling Factor:1"
df = df[~df[target_col].isna()]

# Split data
train = df[df['YearNum'].isin([2014, 2015, 2016])].copy()
test = df[df['YearNum'] == 2017].copy()

# Features
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

# Preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['State', 'Gender'])
    ],
    remainder='passthrough'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train and predict
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

# Create results DataFrame
results = test[['State', 'Gender', 'Year', target_col]].copy()
results['Predicted'] = y_pred
results['Error'] = results[target_col] - results['Predicted']

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print all results sorted by absolute error
pd.set_option('display.max_rows', None)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")
print("\nAll Predictions vs Actual (Sorted by Error Magnitude):")
print(results[['State', 'Gender', 'Year', target_col, 'Predicted', 'Error']]
      .sort_values('Error', key=np.abs, ascending=False)
      .to_string(index=False))
pd.reset_option('display.max_rows')

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Dropout Rate (2017)")
plt.ylabel("Predicted Dropout Rate (2017)")
plt.title("Actual vs Predicted Secondary Dropout Rate (2017)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Save results
results.to_csv('secondary_dropout_2017_predictions_vs_actual.csv', index=False)
