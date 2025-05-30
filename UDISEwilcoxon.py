import pandas as pd
from scipy.stats import wilcoxon

# Load data
file_path = 'eesaahmed197_17484137453452513.csv'
df = pd.read_csv(file_path)

# Extract year as integer
df['Year'] = df['Year'].str.extract(r'(\d{4})').astype(int)

# Filter relevant columns
target_col = "Drop-Out Rate Of Secondary School (UOM:Ratio), Scaling Factor:1"
df = df[['State', 'Gender', 'Year', target_col]].dropna()

# Compare consecutive years using Wilcoxon test
years = sorted(df['Year'].unique())
for i in range(1, len(years)):
    year_prev = years[i-1]
    year_current = years[i]
    
    # Get paired data for states present in both years
    prev_data = df[df['Year'] == year_prev].set_index(['State', 'Gender'])[target_col]
    current_data = df[df['Year'] == year_current].set_index(['State', 'Gender'])[target_col]
    paired_data = prev_data.align(current_data, join='inner')
    
    # Run Wilcoxon test
    stat, p = wilcoxon(paired_data[0], paired_data[1])
    print(f"Wilcoxon test between {year_prev} and {year_current}:")
    print(f"  Statistic = {stat:.2f}, p-value = {p:.4f}")
    if p < 0.05:
        print("  Significant difference (reject H₀)")
    else:
        print("  No significant difference (fail to reject H₀)")
