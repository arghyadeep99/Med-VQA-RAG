import pandas as pd
import numpy as np

disease_df = pd.read_csv("disease_features.csv", header="infer", sep=",", encoding="utf-8", dtype=str, keep_default_na=False)
drugs_df = pd.read_csv("drug_features.csv", header="infer", sep=",", encoding="utf-8", dtype=str, keep_default_na=False)

print(disease_df.columns)
print(drugs_df.columns)

disease_df_cols = ["mondo_name", "mondo_definition", "umls_description", "orphanet_definition", 
                    "orphanet_prevalence", "orphanet_epidemiology", "orphanet_clinical_description", 
                    "orphanet_management_and_treatment", "mayo_symptoms", "mayo_causes", "mayo_risk_factors",
                    "mayo_complications", "mayo_prevention", "mayo_see_doc"]

drugs_df_cols = ["indication", "mechanism_of_action", "pharmacodynamics", "protein_binding"]

disease_df[disease_df_cols] = disease_df[disease_df_cols].fillna("").astype(str)
drugs_df[drugs_df_cols] = drugs_df[drugs_df_cols].fillna("").astype(str)

disease_df["combined_text"] = disease_df[disease_df_cols].agg(' '.join, axis=1).str.strip()
drugs_df["combined_text"] = drugs_df[drugs_df_cols].agg('. '.join, axis=1).str.strip()

disease_df["combined_text"] = disease_df["combined_text"].replace('', pd.NA)
drugs_df["combined_text"] = drugs_df["combined_text"].replace('', pd.NA)

disease_df["word_count"] = disease_df["combined_text"].str.split().str.len()
drugs_df["word_count"] = drugs_df["combined_text"].str.split().str.len()


print(f"Total word count in disease_df: {disease_df['word_count'].sum()}")
print(f"Total word count in drugs_df: {drugs_df['word_count'].sum()}")


disease_df.to_csv("disease_features_combined.csv", sep=',', encoding='utf-8', index=False)
drugs_df.to_csv("drug_features_combined.csv", sep=',', encoding='utf-8', index=False)