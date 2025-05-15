
import pandas as pd
import numpy as np

nodes_df = pd.read_csv("nodes.csv", header="infer", sep=",", encoding="utf-8", dtype=str, keep_default_na=False)
edges_df = pd.read_csv("edges2.csv", header="infer", sep=",", encoding="utf-8", dtype=str, keep_default_na=False)

preserve_edges = ["disease_protein", "contraindication", 
 "drug_protein", "off-label use", "indication", "disease_phenotype_positive", "disease_phenotype_negative", "exposure_disease",
  "drug_effect"] # "anatomy_protein_absent", "anatomy_protein_present", "bioprocess_protein", "exposure_protein", "protein_protein", "drug_drug", 

filtered_edges_df = edges_df[edges_df[':TYPE'].isin(preserve_edges)].copy()

connected_node_ids = pd.unique(filtered_edges_df[[":START_ID", ":END_ID"]].values.ravel())

filtered_nodes_df = nodes_df[nodes_df['node_index:ID'].isin(connected_node_ids)].copy()

print(len(filtered_edges_df))
print(len(filtered_nodes_df))

filtered_edges_df.to_csv("filtered_edges.csv", sep=',', encoding='utf-8', index=False)
filtered_nodes_df.to_csv("filtered_nodes.csv", sep=',', encoding='utf-8', index=False)
