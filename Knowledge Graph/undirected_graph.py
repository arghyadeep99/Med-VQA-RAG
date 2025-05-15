import pandas as pd
df = pd.read_csv(r"edges.csv", header="infer", sep=",", encoding="utf-8", dtype=str, keep_default_na=False)
                 
group = df[[":START_ID", ":END_ID", ":TYPE", "display_relation"]].agg(frozenset, axis=1)
df_result = df.groupby(group).first()
df_result.to_csv(r"edges2.csv", sep=',', encoding='utf-8', index=False)