import os
import sys
import json
import faiss
import numpy as np
import torch
from PIL import Image
from utils.embedding_model import EmbeddingModelLoader
from data.data_loader import DataLoader

class Indexer:
    def __init__(self, 
                model,
                preprocess,
                tokenizer,
                context_length,
                index_file, 
                doc_map_file = './image_report_mapping.json',
                vector_db: str = "faiss"
                ):
        
        self.vector_db = vector_db
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.index_file = index_file
        self.doc_map_file = doc_map_file

        with open(self.doc_map_file, "r", encoding="utf-8") as f:
            self.doc_map = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device).eval()

    def _load_index(self):
        #TODO: add different vectordb
        if self.vector_db == "faiss":
            self.index = faiss.read_index(self.index_file)

    def _build_index(self):
        #TODO: add different vectordb
        if self.vector_db == "faiss":
            embs = []

            for item in self.doc_map:
                report_path  = item.get("report_path", "")
                image_path = item.get("image_path", "")
                if not report_path or not image_path:
                    continue

                emb = self._compute_embedding(report_path, image_path)
                embs.append(emb)

            embs = np.stack(embs).astype("float32")
            dim  = embs.shape[1]
            idx  = faiss.IndexFlatL2(dim)
            idx.add(embs)

            # persist
            faiss.write_index(idx, self.index_file)
            self.index = idx
        
    def _compute_embedding(self, report_path, image_path):
        # load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_input = self.preprocess(img).unsqueeze(0).to(self.device)

        # tokenize text
        with open(report_path, "r") as f:
            text = f.read()

        txt_input = self.tokenizer([text], context_length=self.context_length).to(self.device)

        # forward pass
        with torch.no_grad():
            img_feats, txt_feats, _ = self.model(img_input, txt_input)

        joint = (img_feats + txt_feats) / 2.0
        return joint.cpu().numpy()[0]  
    
def main():

    model_name = "biomedclip" #TODO: add different models
    index_file_name = "./biomedclip_index.index"
    doc_map = "./image_report_mapping.json"
    vector_db = "faiss"

    model_loader = EmbeddingModelLoader(model_name=model_name)
    emb_model, preprocess, emb_tokenizer, context_length = model_loader.load_model_and_tokenizer()

    indexer = Indexer(emb_model, preprocess, emb_tokenizer, context_length, index_file_name, doc_map, vector_db)

    if os.path.exists(indexer.index_file):
        indexer._load_index()

    else:
        indexer._build_index()

if __name__ == "__main__":
    main()