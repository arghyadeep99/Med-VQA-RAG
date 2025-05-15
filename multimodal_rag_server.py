from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from utils.embedding_model import EmbeddingModelLoader
from modules.retriever import Retriever
from modules.indexer import Indexer

app = FastAPI(title="MultiModal RAG Retrieval API")

# --- CONFIGURATION ---
MODEL_NAME = "biomedclip"
INDEX_DIR = "./datasets/indexed_files/"
INDEX_FILE = os.path.join(INDEX_DIR, "biomedclip_index_merged.index")
DOC_MAP = "./image_report_mapping_all_227835.json"
VECTOR_DB = "faiss"
TOP_K = 5  # default number of results


class RetrieveRequest(BaseModel):
    image_path: str
    user_query: str
    top_k: int = TOP_K


@app.on_event("startup")
def load_resources():
    global indexer, retriever_base

    # 1. Load embedding model & tokenizer
    loader = EmbeddingModelLoader(model_name=MODEL_NAME)
    emb_model, preprocess, emb_tokenizer, context_length = loader.load_model_and_tokenizer()

    # 2. Initialize Indexer and ensure FAISS index is built/loaded
    indexer = Indexer(
        emb_model, preprocess, emb_tokenizer, context_length,
        INDEX_FILE, DOC_MAP, VECTOR_DB
    )
    if os.path.isdir(INDEX_DIR) and os.listdir(INDEX_DIR):
        if os.path.exists(INDEX_FILE):
            indexer._load_index()
        else:
            indexer.merge_faiss_indexes(INDEX_DIR, INDEX_FILE)
            indexer._load_index()
    else:
        indexer._build_index()


@app.post("/retrieve/")
def retrieve(request: RetrieveRequest):
    """
    Given an image_path and a text query, return the top-k similar reports.
    """
    if not os.path.isfile(request.image_path):
        raise HTTPException(status_code=400, detail="Image path does not exist")

    # 3. Perform retrieval
    retriever = Retriever(
        indexer,
        request.image_path,
        request.user_query,
        top_k=request.top_k,
        metric="cosine"
    )
    results = retriever.retrieve_similar_items()

    # 4. Read the report contents and assemble the response
    response = []
    for r in results:
        # Assuming report_path is relative or absolute path on disk
        report_full_path = os.path.join("./all_reports", os.path.basename(r["report_path"]))
        if not os.path.isfile(report_full_path):
            # skip or return placeholder
            report_text = ""
        else:
            with open(report_full_path, "r") as f:
                report_text = f.read()

        response.append({
            "rank": r["rank"],
            "image": r["image_path"],
            "report": r["report_path"],
            "content": report_text,
            "score": round(r["score"], 4),
        })

    return {"query": request.user_query, "results": response}
