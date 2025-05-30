import pandas as pd
file_path = 'eesaahmed197_17485821455420601.csv'
df = pd.read_csv(file_path)
def extract_year(year_str):
    try:
        return int(str(year_str).split(',')[-1].strip())
    except Exception:
        return None

df['YearNum'] = df['Year'].apply(extract_year)


qualification_columns_highest = [
    "Percentage Distribution Of Teachers Whose Academic Qualification Is Below Secondary (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Percentage Distribution Of Teachers Whose Academic Qualification Is Secondary (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Percentage Distribution Of Teachers Whose Academic Qualification Is Higher Secondary (%) (UOM:%(Percentage)), Scaling Factor:1"
]

qualification_columns_least = [
    "Percentage Distribution Of Teachers Whose Academic Qualification Is Graduate (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Percentage Distribution Of Teachers Whose Academic Qualification Is Post Graduate (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Percentage Distribution Of Teachers Whose Academic Qualification Is Mphil (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Percentage Distribution Of Teachers Whose Academic Qualification Is Phd/Post Doctoral (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Percentage Distribution Of Teachers Who Are Under State Defined Academic Qualification (%) (UOM:%(Percentage)), Scaling Factor:1",
    "Percentage Distribution Of Teachers Under No Response (%) (UOM:%(Percentage)), Scaling Factor:1"
]

cols_to_check = ['State', 'YearNum'] + qualification_columns_highest + qualification_columns_least
df_clean = df.dropna(subset=cols_to_check)

print("\n--- Top 5 states with HIGHEST ratios up to and including Higher Secondary qualification ---")
for col in qualification_columns_highest:
    print(f"\nColumn: {col}")
    for year in sorted(df_clean['YearNum'].unique()):
        temp = df_clean[df_clean['YearNum'] == year][['State', col]]
        if temp.empty:
            continue
        top5 = temp.nlargest(5, col)
        print(f"Year: {year}")
        print(top5.to_string(index=False))

print("\n--- Top 5 states with LEAST ratios after Higher Secondary qualification ---")
for col in qualification_columns_least:
    print(f"\nColumn: {col}")
    for year in sorted(df_clean['YearNum'].unique()):
        temp = df_clean[df_clean['YearNum'] == year][['State', col]]
        if temp.empty:
            continue
        top5 = temp.nsmallest(5, col)
        print(f"Year: {year}")
        print(top5.to_string(index=False))
