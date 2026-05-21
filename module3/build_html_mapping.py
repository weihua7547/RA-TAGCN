import os
import json
import pickle
import numpy as np
from tqdm import tqdm

# ========= 路徑設定（只指定單一 JSONL） =========
JSONL_PATH = "module1/data_preprocessing/output/finetune/vectordb.jsonl"
EMB_PATH   = "module3/db_data/html_embeddings/vectordb.npy"
OUT_PATH   = "module3/db_data/id_mapping.pkl"

assert os.path.exists(JSONL_PATH), "❌ JSONL_PATH 不存在"
assert os.path.exists(EMB_PATH),   "❌ EMB_PATH 不存在"

mapping = []
global_id = 0

print("🔄 Rebuilding ID mapping (single JSONL)...")

# ---------- 讀 embedding 數量（mmap，不吃 RAM） ----------
vecs = np.load(EMB_PATH, mmap_mode="r")
n_vecs = vecs.shape[0]

# ---------- 對應 JSONL 行號 ----------
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line_no in tqdm(range(n_vecs), desc="Mapping vectors"):
        mapping.append((os.path.basename(JSONL_PATH), line_no))
        global_id += 1

print(f"📦 Total vectors mapped: {global_id}")

# ---------- 儲存 mapping ----------
with open(OUT_PATH, "wb") as f:
    pickle.dump(mapping, f)

print(f"✅ Mapping saved to {OUT_PATH}")
