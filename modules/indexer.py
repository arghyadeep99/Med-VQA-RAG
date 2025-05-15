import os
import re
import sys
import json
import faiss
import numpy as np
import torch
from PIL import Image
from utils.embedding_model import EmbeddingModelLoader
from tqdm import tqdm
from scipy import spatial

def natural_sort(file_list):
    """
    Sort file list in natural numerical order.
    """
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    return sorted(file_list, key=natural_keys)

class Indexer:
    def __init__(self, 
                model,
                preprocess,
                tokenizer,
                context_length,
                index_file, 
                doc_map_file = "./image_report_mapping_all_227835.json",
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
        print(f"Loaded index...")

    def _build_index(self, checkpoint_interval=5000):
        resume_from = 150000
        if self.vector_db == "faiss":
            embs = []
            dim = None
            count = 0

            print(f"Resuming FAISS index from checkpoint {resume_from}...")

            for i, item in enumerate(tqdm(self.doc_map, desc="Embedding items")):
                if i < resume_from:
                    continue  # Skip items already embedded

                report_path = item.get("report_path", "")
                image_path = item.get("image_path", "")
                if not report_path or not image_path:
                    continue

                try:
                    emb = self._compute_embedding(report_path, image_path)
                    embs.append(emb)
                    count += 1
                except Exception as e:
                    print(f"Failed to compute embedding for {image_path}: {e}")
                    continue

                if count % checkpoint_interval == 0:
                    embs_np = np.stack(embs).astype("float32")
                    faiss.normalize_L2(embs_np)

                    # Create a fresh index for each chunk
                    dim = embs_np.shape[1]
                    idx = faiss.IndexHNSWFlat(dim, 32)
                    idx.metric_type = faiss.METRIC_INNER_PRODUCT

                    idx.add(embs_np)
                    embs = []  # Clear buffer

                    total_count = resume_from+count

                    step_path = self.index_file.replace(".index", f"_step_{total_count}.index")
                    print(f"Saving checkpoint index at {step_path}")
                    faiss.write_index(idx, step_path)

                    # Completely discard index from memory
                    del idx

            # Final flush
            if embs:
                embs_np = np.stack(embs).astype("float32")
                faiss.normalize_L2(embs_np)

                dim = embs_np.shape[1]
                idx = faiss.IndexHNSWFlat(dim, 32)
                idx.metric_type = faiss.METRIC_INNER_PRODUCT

                idx.add(embs_np)

                final_path = self.index_file.replace(".index", f"_step_{resume_from + count}.index")
                print(f"Saving final checkpoint index at {final_path}")
                faiss.write_index(idx, final_path)
                del idx

            print("All checkpoints saved. Merge them later.")

    def merge_faiss_indexes(self, index_dir, output_path):
        """
        Incrementally merges multiple FAISS HNSW index files into one final index.
        """
        index_paths = natural_sort([
            os.path.join(index_dir, f)
            for f in os.listdir(index_dir)
            if f.endswith(".index")
        ])

        if not index_paths:
            raise ValueError("No index paths provided for merging.")

        print(f"Merging {len(index_paths)} FAISS indexes...")

        print(f"Loading: {index_paths[0]}")
        base_index = faiss.read_index(index_paths[0])

        dim = base_index.d # Create a fresh index with the same dimensionality and M
        M = 32  # Same M used during indexing
        merged_index = faiss.IndexHNSWFlat(dim, M)
        merged_index.metric_type = faiss.METRIC_INNER_PRODUCT

        # Add vectors from each index file
        for path in index_paths:
            print(f"Merging: {path}")
            index = faiss.read_index(path)
            merged_index.add(index.reconstruct_n(0, index.ntotal))  # Add all vectors
            del index  # Free memory


        print(f"Merged index saved to: {output_path}")
        faiss.write_index(merged_index, output_path)
        print("Merging complete.")

        return merged_index

    def _compute_embedding(self, user_query, image_path):
        # load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_input = self.preprocess(img).unsqueeze(0).to(self.device)

        # # tokenize text
        # with open(user_query, "r") as f:
        #     text = f.read()

        txt_input = self.tokenizer([user_query], context_length=self.context_length).to(self.device)

        # forward pass
        with torch.no_grad():
            img_feats, txt_feats, _ = self.model(img_input, txt_input)

        joint = (img_feats + txt_feats) / 2.0
        return joint.cpu().numpy()[0]  