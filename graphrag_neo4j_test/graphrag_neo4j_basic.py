#!/usr/bin/env python3
import os
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

# ─── Configuration ─────────────────────────────────────────────────────────────
NEO4J_URI = "neo4j+s://9301fe45.databases.neo4j.io"  # Bolt routing URI
NEO4J_USER = "neo4j"  # AuraDB username
NEO4J_PASSWORD = "l_jBXBupg2kTYC7kdbcUdf4aJ2Pc8L5XwBxGA09m8tY"  # AuraDB password
VECTOR_INDEX = "idx_desc_embedding_Disease"  # Prebuilt vector index name
MODEL_NAME = "gpt-4o-mini"  # Pin to gpt-4o-mini
QUERY_TEXT = """Is there any issue with the given chest X Ray results:

Closest Report Findings: 
                                 FINAL REPORT
 EXAMINATION:  CHEST (PORTABLE AP)
 
 INDICATION:  History: ___F with fall from standing. Reports headache, neck
 pain, Tspine TTP, right shoulder TTP  // eval for ICH, spinal fracture,
 shoulder fracture/fracture
 
 COMPARISON:  CT T-spine from earlier the same day and chest radiographs ___
 
 FINDINGS: 
 
 AP semi upright view of the chest provided.
 
 There is no focal consolidation, effusion, or pneumothorax.  Bibasilar
 atelectasis is similar to prior.  Mild cardiomegaly and large hiatal hernia
 are similar to prior. Imaged osseous structures are intact.  No free air below
 the right hemidiaphragm is seen.
 
 IMPRESSION: 
 
 No acute intrathoracic process.
 
"""

# Load API key from environment (required)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")

# ─── Step 1: Connect to AuraDB ─────────────────────────────────────────────────
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    # encrypted=True
)

# ─── Step 2: Initialize Embedder & Retriever ────────────────────────────────────
# embedder = OpenAIEmbeddings(model="text-embedding-3-large")

# explicitly request the 1536-d embedding model
embedder = OpenAIEmbeddings(
    model="text-embedding-3-small",   # ← 1536 dimensions
    api_key=OPENAI_API_KEY
)

retriever = VectorRetriever(
    driver=driver,
    index_name=VECTOR_INDEX,
    embedder=embedder
)

# ─── Step 3: Initialize LLM ────────────────────────────────────────────────────
llm = OpenAILLM(
    model_name=MODEL_NAME,
    model_params={"temperature": 0},
    api_key=OPENAI_API_KEY
)

# ─── Step 4: Build GraphRAG Pipeline ────────────────────────────────────────────
rag = GraphRAG(
    retriever=retriever,
    llm=llm
)

# ─── Step 5: Execute the RAG Query ─────────────────────────────────────────────
result = rag.search(
    query_text=QUERY_TEXT,
    retriever_config={"top_k": 5}
)

# ─── Output ───────────────────────────────────────────────────────────────────
print("Answer:\n", result.answer)
