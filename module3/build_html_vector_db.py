import os
import json
import gc
import numpy as np
import faiss
from tqdm import tqdm
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import torch

# ========= 系統與效能設定 =========
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "24"
os.environ["MKL_NUM_THREADS"] = "12"
torch.set_num_threads(12)
torch.set_num_interop_threads(12)

# ========= 路徑設定（只指定單一 JSONL） =========
JSONL_PATH = "module1/data_preprocessing/output/finetune/vectordb.jsonl"
EMB_DIR = "module3/db_data/html_embeddings"
INDEX_PATH = "module3/db_data/html.faiss"  # 建議改名避免覆蓋舊 index
os.makedirs(EMB_DIR, exist_ok=True)

# ========= 載入 Embedding 模型 =========
print("🔄 Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
DIM = 384  # bge-small-en-v1.5 的 embedding 維度也是 384

# ========= HTML → 結構化文字 =========
def html_to_struct_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
        # 1️⃣ DOM tag sequence
        tags = [tag.name for tag in soup.find_all()]
        tag_text = " ".join(tags[:200])
        # 2️⃣ Visible text
        visible_text = soup.get_text(" ", strip=True)[:500]
        # 3️⃣ Structural statistics
        stats_text = (
            f"links={len(soup.find_all('a'))} "
            f"forms={len(soup.find_all('form'))} "
            f"scripts={len(soup.find_all('script'))} "
            f"iframes={len(soup.find_all('iframe'))} "
            f"passwords={len(soup.find_all('input', {'type': 'password'}))}"
        )
        return f"{tag_text} [SEP] {visible_text} [SEP] {stats_text}"
    except Exception:
        return ""

# ========= Step 1: 產生 Embedding =========
print("\n=== Step 1: Generating Embeddings ===")
fname = os.path.basename(JSONL_PATH)
emb_path = os.path.join(EMB_DIR, fname.replace(".jsonl", "_bge-small.npy"))  # 建議加後綴避免覆蓋
if os.path.exists(emb_path):
    print(f"✔ Embedding already exists: {emb_path}")
else:
    texts = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading HTML"):
            try:
                obj = json.loads(line)
                struct_text = html_to_struct_text(obj.get("html", ""))
                if struct_text:
                    texts.append(struct_text)
            except Exception:
                continue
    
    if len(texts) == 0:
        raise RuntimeError("❌ No usable HTML found in the JSONL file.")
    
    print(f"🧠 Encoding {len(texts)} samples...")
    all_embeddings = []
    chunk_size = 256  # 控制 RAM 使用量，3090 可調高到 512 或 1024
    for i in tqdm(range(0, len(texts), chunk_size), desc="Encoding"):
        batch_texts = texts[i:i + chunk_size]
        emb = model.encode(
            batch_texts,
            batch_size=64,              # 3090 可開大一點，加速很多
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True   # bge 系列也需要 normalize 才能用 IP
        )
        all_embeddings.append(emb)
        # 每 chunk 清一次記憶體
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    embeddings = np.vstack(all_embeddings)
    np.save(emb_path, embeddings)
    print(f"💾 Saved embeddings to {emb_path}")
    
    del texts, embeddings, all_embeddings
    gc.collect()

# ========= Step 2: 建立 FAISS Index =========
print("\n=== Step 2: Building FAISS Index ===")
index = faiss.IndexFlatIP(DIM)  # cosine similarity（因為 normalize 了）
vecs = np.load(emb_path)
index.add(vecs)
print(f"📦 Total vectors in index: {index.ntotal}")
faiss.write_index(index, INDEX_PATH)
print(f"\n✅ FAISS index saved to: {INDEX_PATH}")