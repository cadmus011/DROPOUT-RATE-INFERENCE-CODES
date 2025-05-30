import pandas as pd
file_path = 'eesaahmed197_17484137453452513.csv'
df = pd.read_csv(file_path)
def extract_year(year_str):
    try:
        return int(str(year_str).split(',')[-1].strip())
    except Exception:
        return None

df['YearNum'] = df['Year'].apply(extract_year)
df = df.dropna(subset=['YearNum'])
df['YearNum'] = df['YearNum'].astype(int)

education_columns = [
    "Drop-Out Rate Of Primary School (UOM:Ratio), Scaling Factor:1",
    "Drop-Out Rate Of Upper Primary School (UOM:Ratio), Scaling Factor:1",
    "Drop-Out Rate Of Elementary School (UOM:Ratio), Scaling Factor:1",
    "Drop-Out Rate Of Secondary School (UOM:Ratio), Scaling Factor:1",
    "Drop-Out Rate Of Higher Secondary School (Grade Xi To Xii) (UOM:Ratio), Scaling Factor:1"
]
results = []
for year in sorted(df['YearNum'].unique()):
    for col in education_columns:
        temp = df[(df['YearNum'] == year) & (df['Gender'].isin(['Boys', 'Girls']))]
        pivot = temp.pivot(index='State', columns='Gender', values=col)

        pivot = pivot.dropna(subset=['Boys', 'Girls'])
        if pivot.empty:
            continue
        pivot['GenderGap'] = (pivot['Boys'] - pivot['Girls']).abs()

        max_gap = pivot['GenderGap'].max()
        max_states = pivot[pivot['GenderGap'] == max_gap]
        for state, row in max_states.iterrows():
            results.append({
                'Year': year,
                'Education Level': col,
                'State': state,
                'Boys Rate': row['Boys'],
                'Girls Rate': row['Girls'],
                'Gender Gap': row['GenderGap']
            })

results_df = pd.DataFrame(results)
print("State(s) with Highest Gender Gap for Each Year and Education Level:")
print(results_df.to_string(index=False))


