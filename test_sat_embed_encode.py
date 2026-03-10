import sys, os
import faiss
import numpy as np

idx_path = "sat_medical_cache_sat_trained/faiss.index"
if os.path.exists(idx_path):
    index = faiss.read_index(idx_path)
    print("FAISS Index dim:", index.d)
    print("FAISS Index vectors:", index.ntotal)
    
emb_path = "sat_medical_cache_sat_trained/embeddings.npy" 
if os.path.exists(emb_path):
    embeds = np.load(emb_path)
    print("Embeddings shape:", embeds.shape)
