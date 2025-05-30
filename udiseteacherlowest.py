import pandas as pd
file_path = 'eesaahmed197_17485821455420601.csv'
df = pd.read_csv(file_path)
def extract_year(year_str):
    try:
        return int(str(year_str).split(',')[-1].strip())
    except Exception:
        return None

df['YearNum'] = df['Year'].apply(extract_year)
qualification_columns = [col for col in df.columns if 'Percentage Distribution Of Teachers Whose Academic Qualification' in col]

for col in qualification_columns:
    print(f"\n=== Top 10 states with least ratio for: {col} ===")
    for year in sorted(df['YearNum'].dropna().unique()):
        temp = df[df['YearNum'] == year][['State', col]].dropna()
        top10 = temp.nsmallest(10, col)
        print(f"\nYear: {year}")
        print(top10.to_string(index=False))

