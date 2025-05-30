import pandas as pd

# Load dataset
df = pd.read_csv('eesaahmed197_17484137453452513.csv')

# Extract year from string
df['Year'] = df['Year'].str.extract(r'(\d{4})').astype(int)

# Column name mapping
edu_map = {
    'Drop-Out Rate Of Primary School (UOM:Ratio), Scaling Factor:1': 'Primary',
    'Drop-Out Rate Of Upper Primary School (UOM:Ratio), Scaling Factor:1': 'UpperPrimary',
    'Drop-Out Rate Of Elementary School (UOM:Ratio), Scaling Factor:1': 'Elementary',
    'Drop-Out Rate Of Secondary School (UOM:Ratio), Scaling Factor:1': 'Secondary',
    'Drop-Out Rate Of Higher Secondary School (Grade Xi To Xii) (UOM:Ratio), Scaling Factor:1': 'HigherSecondary'
}

results = []

# Analyze each education level
for full_col, short_col in edu_map.items():
    # Process each year and gender combination
    for (year, gender), group in df.groupby(['Year', 'Gender']):
        temp = group[['State', full_col]].dropna()
        if temp.empty:
            continue
        
        # Find minimum dropout rate
        min_rate = temp[full_col].min()
        min_states = temp[temp[full_col] == min_rate]['State'].tolist()
        
        results.append({
            'Year': year,
            'EducationLevel': short_col,
            'Gender': gender,
            'MinRate': min_rate,
            'States': min_states
        })

# Print formatted results
for entry in results:
    states = ', '.join(entry['States'])
    print(f"Year: {entry['Year']} | Education: {entry['EducationLevel']:>13} | Gender: {entry['Gender']:>6} | Lowest Rate: {entry['MinRate']} | States: {states}")
