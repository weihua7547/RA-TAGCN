import os
import json
import logging
import pickle
import torch
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from preprocessor import html_to_graph_data

# ========= 路徑與參數設定 =========
TRAIN_JSONL = "module1/data_preprocessing/output/finetune/train.jsonl"
FAISS_INDEX_PATH = "module3/db_data/html.faiss"
ID_MAPPING_PATH = "module3/db_data/id_mapping.pkl"
JSONL_PATH = "module1/data_preprocessing/output/finetune/vectordb.jsonl"
OUTPUT_PATH = "module3/db_data/rac_train_pairs.jsonl"

TOP_K = 50           # 增加候選池，確保能過濾出足夠的優質鄰居
NEIGHBOR_N = 5       # 目標鄰居數
MIN_SIM = 0.65       # 最低相似度門檻
BATCH_SIZE = 32      # 如果電腦容易重啟，建議調低此數值 (原為 64)
WRITE_BUFFER_SIZE = 100  # 每 100 筆資料寫入一次硬碟，減少磁碟 I/O 負擔

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("rac_processing.log")]
    )

setup_logger()
logger = logging.getLogger(__name__)

def html_to_graph_safe(html):
    if not html: return False
    try:
        g = html_to_graph_data(html)
        return g is not None and getattr(g, "x", None) is not None and len(g.x) > 0
    except Exception:
        return False

# ========= 1. 載入資料 (優化記憶體) =========
def load_all_htmls(path):
    print(f"🔹 Loading database HTMLs into RAM...")
    html_pool = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            try:
                obj = json.loads(line)
                html_pool.append(obj.get("html", ""))
            except:
                html_pool.append("")
    
    # 簡單估算記憶體佔用
    import sys
    pool_size_mb = sys.getsizeof(html_pool) / (1024 * 1024)
    print(f"💡 HTML Pool size in RAM: ~{pool_size_mb:.2f} MB")
    return html_pool

# ========= 2. 初始化模型與索引 =========
print("🔹 Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("🔹 Loading ID mapping...")
with open(ID_MAPPING_PATH, "rb") as f:
    id_mapping = pickle.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Loading embedding model on {device}...")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

db_html_pool = load_all_htmls(JSONL_PATH)

# ========= 3. 批次處理主流程 =========
print(f"🚀 Building RAC dataset (Batch: {BATCH_SIZE}, Buffer: {WRITE_BUFFER_SIZE})...")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(TRAIN_JSONL, "r", encoding="utf-8") as fin:
    all_lines = fin.readlines()

total = len(all_lines)
kept = 0
no_neighbor_count = 0
write_buffer = []

# 清空舊檔案
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    pass

for i in tqdm(range(0, total, BATCH_SIZE), desc="Processing Batches"):
    batch_lines = all_lines[i : i + BATCH_SIZE]
    batch_objs = []
    batch_queries = []
    
    # 1. 預處理 Batch
    for line in batch_lines:
        try:
            obj = json.loads(line)
            # 只有當 Query HTML 本身合法時才處理
            if "html" in obj and html_to_graph_safe(obj["html"]):
                batch_objs.append(obj)
                batch_queries.append(obj["html"])
        except Exception as e:
            continue
    
    if not batch_queries:
        continue
        
    try:
        # 2. 批量向量化與檢索
        embeddings = embedder.encode(batch_queries, normalize_embeddings=True, show_progress_bar=False)
        distances, indices = index.search(embeddings, TOP_K)
        
        # 3. 處理檢索結果
        for idx, (scores, nids) in enumerate(zip(distances, indices)):
            valid_candidates = []
            
            for score, nid in zip(scores, nids):
                if nid == -1 or score < MIN_SIM:
                    continue
                
                neighbor_html = db_html_pool[int(nid)]
                if html_to_graph_safe(neighbor_html):
                    # 儲存 (分數, HTML) 方便後續排序
                    valid_candidates.append((float(score), neighbor_html))
                
                # 收集足夠的數量就可以停止檢查這個 Query 的候選池
                if len(valid_candidates) >= NEIGHBOR_N:
                    break
            
            # --- 排序邏輯：按相似度分數由高到低排序 ---
            valid_candidates.sort(key=lambda x: x[0], reverse=True)
            
            if len(valid_candidates) < NEIGHBOR_N:
                no_neighbor_count += 1
            
            # 構建輸出物件
            out_data = batch_objs[idx].copy()
            out_data["query_html"] = out_data.pop("html")
            out_data["neighbor_htmls"] = [item[1] for item in valid_candidates]
            out_data["neighbor_scores"] = [item[0] for item in valid_candidates]
            
            # 加入緩衝區
            write_buffer.append(json.dumps(out_data, ensure_ascii=False) + "\n")
            kept += 1
            
            # --- 磁碟寫入邏輯：累積到一定數量才寫入 ---
            if len(write_buffer) >= WRITE_BUFFER_SIZE:
                with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:
                    fout.writelines(write_buffer)
                write_buffer = []
                
    except Exception as e:
        logger.error(f"Error processing batch starting at index {i}: {e}")
        continue

# 處理最後剩餘的緩衝區
if write_buffer:
    with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:
        fout.writelines(write_buffer)

print(f"\n✅ All tasks completed!")
print(f"Total: {total} | Successfully Kept: {kept} | No-neighbor (<{NEIGHBOR_N}): {no_neighbor_count}")