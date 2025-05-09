import os
from utils.embedding_model import EmbeddingModelLoader
from modules.retriever import Retriever
from modules.indexer import Indexer

def main():

    model_name = "biomedclip" #TODO: add different models
    index_file_dir = "./datasets/indexed_files/"
    index_file_name = "./datasets/indexed_files/biomedclip_index_merged.index"
    doc_map = "./image_report_mapping_all_227835.json"
    vector_db = "faiss"

    model_loader = EmbeddingModelLoader(model_name=model_name)
    emb_model, preprocess, emb_tokenizer, context_length = model_loader.load_model_and_tokenizer()

    indexer = Indexer(emb_model, preprocess, emb_tokenizer, context_length, index_file_name, doc_map, vector_db)

    if os.listdir(index_file_dir):
        if os.path.exists(index_file_name):
            indexer._load_index()
        else:
            indexer.merge_faiss_indexes(index_file_dir, index_file_name)
            indexer._load_index()
    else:
        indexer._build_index()

    test_image = "./datasets/mimic-cxr-images-512/files/p19/p19000065/s51613820/58f383e7-edcbd8c7-2f6dc2af-eb97ddf1-f7cbc46a.jpg" #TODO: add your path
    test_report = "./datasets/mimic-cxr/files/p19/p19000065/s51613820.txt" #TODO: add your path
    retriever = Retriever(indexer, test_image, test_report, top_k=5, metric="cosine")
    results = retriever.retrieve_similar_items()
    
    for r in results:
        new_report_path = os.path.join('./all_reports', r['report_path'].split('/')[-1])
        with open(new_report_path, 'r') as f:
            report_data = f.read()
        print(f"Rank {r['rank']}:")
        print(f"  Image: {r['image_path']}")
        print(f"  Report: {r['report_path']}")
        print(f"  Report Content: {report_data}") # TODO: Can save this REPORT CONTENT
        print(f"  Score: {r['score']:.4f}\n")

if __name__ == "__main__":
    main()