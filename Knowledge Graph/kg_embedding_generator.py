from neo4j import GraphDatabase
from neo4j_graphrag.generation.graphrag import GraphRAG
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorRetriever, HybridRetriever
from dotenv import load_dotenv
import os

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBED_LABELS = ["disease", "drug"] #"gene__protein", "exposure", "effect__phenotype"

neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
neo4j_session = neo4j_driver.session(database=NEO4J_DATABASE)


def import_batch(driver, nodes, batch_n, label):
    
    records, summary, keys = driver.execute_query("CALL genai.vector.encodeBatch($to_encode_list, 'OpenAI', { token: $token, model: $model }) YIELD index, vector "
    f"MATCH (n:{label} " "{ node_name: $nodes[index].node_name, combined_text: $nodes[index].desc }) "
    "CALL db.create.setNodeVectorProperty(n, 'desc_embedding', vector)",
    nodes=nodes, to_encode_list=[node['to_encode'] for node in nodes], token=OPENAI_API_KEY, model=EMBEDDING_MODEL,
    database_=NEO4J_DATABASE)
    print(f'Processed batch {batch_n}')

    records, _, _ = neo4j_driver.execute_query('''
    MATCH (n:disease|drug WHERE n.desc_embedding IS NOT NULL)
    RETURN count(*) AS countNodesWithEmbeddings, size(n.desc_embedding) AS embeddingSize
    ''', database_=NEO4J_DATABASE)
    print(f"""
    Embeddings generated and attached to nodes.
    Movie nodes with embeddings: {records[0].get('countNodesWithEmbeddings')}.
    Embedding size: {records[0].get('embeddingSize')}.
        """)
    


data_batch = []
batch_n, batch_size = 1, 100

with neo4j_session as session:
    neo4j_driver.verify_connectivity()
    print("Connected to Neo4j database successfully.")
    for label in EMBED_LABELS:
        result = session.run(f"MATCH (n: {label}) RETURN n.node_name AS node_name, n.combined_text AS desc")
        for record in result:
            node_name = record["node_name"]
            desc = record["desc"]
            if node_name and desc:
                data_batch.append({"node_name": node_name, "desc": desc, "to_encode": f"Title: {node_name}, Description: {desc}"})
            
            if len(data_batch) == batch_size:
                import_batch(neo4j_driver, data_batch, batch_n, label)
                data_batch = []
                batch_n += 1

        import_batch(neo4j_driver, data_batch, batch_n, label)
            
        print(f"Completed embedding nodes with label {label}…")


records, _, _ = neo4j_driver.execute_query('''
    MATCH (n:disease|drug WHERE n.desc_embedding IS NOT NULL)
    RETURN count(*) AS countNodesWithEmbeddings, size(n.desc_embedding) AS embeddingSize
    ''', database_=NEO4J_DATABASE)
print(f"""
Embeddings generated and attached to nodes.
Movie nodes with embeddings: {records[0].get('countNodesWithEmbeddings')}.
Embedding size: {records[0].get('embeddingSize')}.
    """)

driver.close()

