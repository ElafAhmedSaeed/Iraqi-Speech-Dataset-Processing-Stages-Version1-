import pandas as pd

df = pd.read_csv("metadata/metadata.csv")

print(df.columns.tolist())