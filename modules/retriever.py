import os
import json
import faiss
import numpy as np

class Retriever:
    def __init__(self, 
                indexer,
                query_image_path, 
                user_query,
                top_k = 1,
                metric = "cosine"):
        
        self.indexer = indexer
        self.query_image_path = query_image_path
        self.user_query = user_query
        self.top_k = top_k
        self.metric = metric

        self.query_embeddings = None
    
    def retrieve_similar_items(self):
        """
        Retrieve top-k most similar entries from the FAISS index given an image + report query.
        """
        # 1. Compute query embedding
        query_embedding = self.indexer._compute_embedding(self.user_query, self.query_image_path)
        query_embedding = query_embedding.astype("float32").reshape(1, -1)

        print(f"Query vector norm: {np.linalg.norm(query_embedding):.4f}")
        print(f"Sample index vector norm: {np.linalg.norm(self.indexer.index.reconstruct(0)):.4f}")


        # 2. Normalize if using cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_embedding)

        # 3. Perform search
        distances, indices = self.indexer.index.search(query_embedding, self.top_k)

        # 4. Map results back to file paths
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            item = self.indexer.doc_map[idx]

            # Use raw dot product score (cosine)
            similarity = dist
            results.append({
                "rank": len(results) + 1,
                "image_path": item.get("image_path", "N/A"),
                "report_path": item.get("report_path", "N/A"),
                "score": float(similarity)
            })

        return results

        
