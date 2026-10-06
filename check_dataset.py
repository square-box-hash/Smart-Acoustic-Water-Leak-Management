import pandas as pd
df = pd.read_pickle("trials.pkl")
print(df.columns.tolist())
print(df.dtypes)
print(df.head(3).to_string())
print(df['leak_type'].unique() if 'leak_type' in df.columns else "no leak_type col")