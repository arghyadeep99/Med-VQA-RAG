# biomed_rag.py

import os
import json
import faiss
import numpy as np
import torch
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

class BiomedRAG:
    def __init__(
        self,
        annotation_file: str = "annotation.json",
        index_file: str      = "biomed_index.index",
        docs_file: str       = "biomed_docs.json",
        model_name: str      = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        context_length: int  = 256,
    ):
        self.annotation_file = annotation_file
        self.index_file      = index_file
        self.docs_file       = docs_file
        self.context_length  = context_length

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load BiomedCLIP
        self.model, self.preprocess = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()

        # load or build index
        if os.path.exists(self.index_file) and os.path.exists(self.docs_file):
            self._load_index()
        else:
            self._build_index()

    def _load_index(self):
        self.index = faiss.read_index(self.index_file)
        with open(self.docs_file, "r", encoding="utf-8") as f:
            self.docs = json.load(f)

    def _build_index(self):
        # read annotation.json
        with open(self.annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f).get("train", [])

        embs = []
        docs = []
        for item in data:
            rpt  = item.get("report", "")
            imgs = item.get("image_path", [])
            if not rpt or not imgs:
                continue

            emb = self._compute_embedding(rpt, imgs[0])
            embs.append(emb)
            docs.append({
                "id":         item.get("id", ""),
                "report":     rpt,
                "image_path": imgs[0]
            })

        embs = np.stack(embs).astype("float32")
        dim  = embs.shape[1]
        idx  = faiss.IndexFlatL2(dim)
        idx.add(embs)

        # persist
        faiss.write_index(idx, self.index_file)
        with open(self.docs_file, "w", encoding="utf-8") as f:
            json.dump(docs, f)

        self.index = idx
        self.docs  = docs

    def _compute_embedding(self, text: str, image_path: str) -> np.ndarray:
        # load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_input = self.preprocess(img).unsqueeze(0).to(self.device)

        # tokenize text
        txt_input = self.tokenizer([text], context_length=self.context_length).to(self.device)

        # forward pass
        with torch.no_grad():
            img_feats, txt_feats, _ = self.model(img_input, txt_input)

        joint = (img_feats + txt_feats) / 2.0
        return joint.cpu().numpy()[0]

    def retrieve(
        self,
        query_text:       str,
        query_image_path: str,
        top_k:            int = 1,
    ) -> list[dict]:
        """
        Returns the top_k documents (dicts with keys 'id','report','image_path')
        most similar to the joint text+image query.
        """
        q_emb = self._compute_embedding(query_text, query_image_path)
        D, I = self.index.search(np.array([q_emb], dtype="float32"), top_k)
        return [self.docs[i] for i in I[0]]


if __name__ == "__main__":
    # simple demo
    retriever = BiomedRAG()
    docs = retriever.retrieve(
        query_text       = "Enlarged heart silhouette with no pleural effusion",
        query_image_path = "CXR1_1_IM-0001//0.png",
        top_k            = 3,
    )
    for d in docs:
        print(f"{d['id']}: {d['report']}")
