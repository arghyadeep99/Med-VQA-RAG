LOAD CSV WITH HEADERS FROM 'file:///drug_features.csv' AS row
MATCH (n {node_index: row.node_index})
SET n.indication = row.indication,
    n.mechanism_of_action = row.mechanism_of_action,
    n.pharmacodynamics = row.pharmacodynamics,
    n.protein_binding = row.protein_binding;

LOAD CSV WITH HEADERS FROM 'file:///disease_features.csv' AS row
MATCH (n {node_index: row.node_index})
SET n.mondo_name = row.mondo_name,
    n.mondo_definition = row.mondo_definition,
        n.umls_description = row.umls_description,
        n.orphanet_description = row.orphanet_definition,
        n.orphanet_prevalence = row.orphanet_prevalence,
        n.orphanet_epidemiology = row.orphanet_epidemiology,
        n.orphanet_clinical_description = row.orphanet_clinical_description,
        n.orphanet_management_and_treatment = row.orphanet_management_and_treatment,
        n.mayo_symptoms = row.mayo_symptoms,
        n.mayo_causes = row.mayo_causes,
        n.mayo_risk_factors = row.mayo_risk_factors,
        n.mayo_complications = row.mayo_complications,
        n.mayo_prevention = row.mayo_prevention,
        n.mayo_see_doc = row.mayo_see_doc;
