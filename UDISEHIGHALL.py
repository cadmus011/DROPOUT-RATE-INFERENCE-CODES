import pandas as pd

# Load the provided CSV file
file_path = 'eesaahmed197_17484137453452513.csv'
df = pd.read_csv(file_path)

# Extract year as integer from the 'Year' column (last 4 digits after comma)
def extract_year(year_str):
    try:
        return int(str(year_str).split(',')[-1].strip())
    except Exception:
        return None

df['YearNum'] = df['Year'].apply(extract_year)
df = df.dropna(subset=['YearNum'])
df['YearNum'] = df['YearNum'].astype(int)

# List of education level dropout columns
education_columns = [
    "Drop-Out Rate Of Primary School (UOM:Ratio), Scaling Factor:1",
    "Drop-Out Rate Of Upper Primary School (UOM:Ratio), Scaling Factor:1",
    "Drop-Out Rate Of Elementary School (UOM:Ratio), Scaling Factor:1",
    "Drop-Out Rate Of Secondary School (UOM:Ratio), Scaling Factor:1",
    "Drop-Out Rate Of Higher Secondary School (Grade Xi To Xii) (UOM:Ratio), Scaling Factor:1"
]

# Filter relevant columns
relevant_cols = ['State', 'Gender', 'YearNum'] + education_columns
df_relevant = df[relevant_cols].dropna()

# For each education column, year, and gender, find top 10 states with highest dropout rate
for col in education_columns:
    for year in sorted(df_relevant['YearNum'].unique()):
        for gender in ['Boys', 'Girls', 'Total']:
            temp = df_relevant[(df_relevant['YearNum'] == year) & (df_relevant['Gender'] == gender)]
            if temp.empty:
                continue
            top10 = temp.nlargest(10, col)[['State', col]]
            print(f"\nTop 10 states for '{col}' dropout rate in {year} for gender {gender}:")
            print(top10.to_string(index=False))


