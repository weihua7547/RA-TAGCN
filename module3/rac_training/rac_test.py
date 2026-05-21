import os
import json
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn import TAGConv, global_mean_pool
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
import warnings
warnings.filterwarnings("ignore", module='bs4')
warnings.filterwarnings("ignore", module='preprocessor')
from preprocessor import html_to_graph_data
import sys
sys.stdout.reconfigure(encoding='utf-8')
import gc

# =========================================================
# Label map（與訓練一致）
# =========================================================
LABEL_MAP = {
    "benign": 0,
    "phishing": 1,
    "defacement": 2,
    "malware": 3,
}
NUM_CLASSES = len(LABEL_MAP)

# =========================================================
# Logging
# =========================================================
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"rac_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return log_path

# =========================================================
# RAC Dataset（與訓練一致）
# =========================================================
class RACGraphDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, label_map=LABEL_MAP, max_neighbors=2):
        self.label_map = label_map
        self.max_neighbors = max_neighbors
        self.samples = []

        # 獲取檔案總大小 (Bytes)
        total_size = os.path.getsize(jsonl_path)
        
        logging.info(f"Loading dataset into RAM: {jsonl_path}")
        logging.info(f"Total file size: {total_size / (1024**3):.2f} GB")

        # 使用 tqdm，設置單位為 Byte (B)，並自動轉換成 KB/MB/GB
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Caching Data (Bytes)") as pbar:
            with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # 記錄當前行的大小並更新進度條
                    line_size = len(line.encode('utf-8', errors='ignore'))
                    
                    try:
                        obj = json.loads(line)
                
                        # 1. 彈性抓取 Query HTML (嘗試多種可能的欄位名稱)
                        q_html = obj.get("query_html") or obj.get("html")
                        
                        # 2. 檢查必要資訊是否存在
                        if q_html is None or "type" not in obj:
                            # 只有在完全沒有 HTML 內容或沒有標籤時才跳過
                            pbar.update(line_size)
                            continue

                        # 轉換 Graph
                        q_graph = html_to_graph_data(q_html)
                        if q_graph is None or torch.isnan(q_graph.x).any():
                            pbar.update(line_size)
                            continue
                        
                        # 3. 鄰居處理 (同樣給予彈性)
                        raw_neighbors = obj.get("neighbor_htmls") or obj.get("neighbors") or []
                        selected_neighbors = raw_neighbors[:self.max_neighbors]
                        
                        neighbors = []
                        for h in selected_neighbors:
                            # 有些數據集的 neighbors 可能是字串也可能是字典，這裡 html_to_graph_data 處理
                            g = html_to_graph_data(h)
                            if g is not None and not torch.isnan(g.x).any():
                                neighbors.append(g)
                        
                        # 填充邏輯 (維持原樣)
                        if not neighbors:
                            neighbors = [q_graph] * self.max_neighbors
                        elif len(neighbors) < self.max_neighbors:
                            neighbors += [neighbors[-1]] * (self.max_neighbors - len(neighbors))
                        
                        label = torch.tensor(self.label_map[obj["type"]], dtype=torch.long)
                        
                        self.samples.append({
                            "query": q_graph,
                            "neighbors": neighbors,
                            "label": label
                        })

                    except json.JSONDecodeError:
                        pass
                    
                    # 更新進度條並定期釋放緩存
                    pbar.update(line_size)
                    if len(self.samples) % 5000 == 0:
                        gc.collect()

        logging.info(f"Successfully cached {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_rac(batch):
    return [b for b in batch if b is not None]

# =========================================================
# TAGCN Encoder（與訓練一致）
# =========================================================
class TagcnEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        self.convs.append(TAGConv(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(TAGConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index):
        if x.std(dim=0).mean() > 0:
            x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        else:
            x = torch.zeros_like(x)
        x = torch.nan_to_num(x, nan=0.0)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# =========================================================
# RAC Graph Classifier with Self-Attention（與訓練一致）
# =========================================================
class RACGraphClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.pool = global_mean_pool
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def encode_graph(self, batch):
        z = self.encoder(batch.x, batch.edge_index)
        g = self.pool(z, batch.batch)
        g = torch.clamp(g, min=-10.0, max=10.0)
        g = torch.nan_to_num(g, nan=0.0)
        return g

    def forward(self, batch_samples, device):
        labels = torch.stack([b["label"] for b in batch_samples]).to(device)
        q_graphs = [b["query"] for b in batch_samples]
        q_batch = PyGBatch.from_data_list(q_graphs).to(device)
        q_emb = self.encode_graph(q_batch)

        max_neighbors = max(len(b["neighbors"]) for b in batch_samples)
        neighbor_graphs = []
        attn_mask = torch.zeros(len(batch_samples), max_neighbors, dtype=torch.bool, device=device)
        for i, b in enumerate(batch_samples):
            ng = b["neighbors"]
            num_ng = len(ng)
            attn_mask[i, :num_ng] = True
            if num_ng < max_neighbors:
                ng += [ng[-1] if ng else b["query"]] * (max_neighbors - num_ng)
            neighbor_graphs.extend(ng)

        n_batch = PyGBatch.from_data_list(neighbor_graphs).to(device)
        n_emb = self.encode_graph(n_batch)
        n_emb = n_emb.view(len(batch_samples), max_neighbors, -1)

        key_padding_mask = ~attn_mask
        attn_out, _ = self.attn(
            q_emb.unsqueeze(1),
            n_emb,
            n_emb,
            key_padding_mask=key_padding_mask
        )
        attn_out = attn_out.squeeze(1)
        attn_out = torch.nan_to_num(attn_out, nan=0.0)

        fused = torch.cat([q_emb, attn_out], dim=1)
        logits = self.classifier(fused)
        return logits, labels

# =========================================================
# Test / Evaluation
# =========================================================
def test(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_pred, all_gt = [], []
    num_samples=0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            if not batch:
                continue
            logits, labels = model(batch, device)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
            total_loss += loss.item() * labels.size(0)
            num_samples += labels.size(0)
            pred = logits.argmax(dim=-1)
            all_pred.extend(pred.cpu().tolist())
            all_gt.extend(labels.cpu().tolist())

    if len(all_gt) == 0:
        logging.info("No valid samples in test set.")
        return

    avg_loss = total_loss / num_samples
    acc = accuracy_score(all_gt, all_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_gt,
        all_pred,
        average="macro",
        zero_division=0
    )

    logging.info("===== TEST RESULT =====")
    logging.info(f"loss : {avg_loss:.4f}")
    logging.info(f"Accuracy : {acc:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall   : {recall:.4f}")
    logging.info(f"F1-score : {f1:.4f}")

    # 完整 classification_report
    inv_label_map = {v: k for k, v in LABEL_MAP.items()}
    report = classification_report(
        all_gt,
        all_pred,
        labels=list(inv_label_map.keys()),
        target_names=list(inv_label_map.values()),
        zero_division=0
    )
    logging.info("\n" + report)

# =========================================================
# Main - Test Pipeline
# =========================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = setup_logger(args.log_dir)
    logging.info(f"Device: {device}")
    logging.info(f"Test jsonl: {args.test_jsonl_path}")
    logging.info(f"Model checkpoint: {args.model_path}")
    logging.info(f"Label map: {LABEL_MAP}")
    # 新增：紀錄使用的 TOP_K 數量
    logging.info(f"Using TOP_K (max_neighbors): {args.top_k}")

    # 修改：將 args.top_k 傳入 Dataset，確保測試時的鄰居數量與訓練邏輯一致
    test_dataset = RACGraphDataset(args.test_jsonl_path, max_neighbors=args.top_k)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0, # 測試時通常不需要太多 worker
        pin_memory=False,
        collate_fn=collate_rac
    )
    logging.info(f"Test samples (after filtering): {len(test_dataset)}")

    # 偵測 input dim
    in_dim = None
    for sample in test_dataset:
        if sample and hasattr(sample["query"], 'x') and sample["query"].x is not None:
            in_dim = sample["query"].x.size(1)
            break
    if in_dim is None:
        raise RuntimeError("Cannot detect node feature dim from test set")
    logging.info(f"Detected node feature dim = {in_dim}")

    # 初始化模型
    encoder = TagcnEncoder(
        in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    model = RACGraphClassifier(
        encoder,
        hidden_dim=args.hidden_dim,
        num_classes=len(LABEL_MAP)
    ).to(device)

    # 載入完整模型 checkpoint
    logging.info(f"Loading model checkpoint: {args.model_path}")
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    logging.info("Model loaded successfully")

    # 執行測試
    test(model, test_loader, device)

    logging.info("Testing finished")
    logging.info(f"Log saved at: {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAC Model Testing Script")
    parser.add_argument("--test_jsonl_path", type=str, default="module3/db_data/rac_test_pairs.jsonl")
    parser.add_argument("--model_path", type=str, default="module3/db_data/checkpoints_rac/DAPT_TAPT/TOP_K_3/rac_best.pt")
    parser.add_argument("--log_dir", type=str, default="module3/db_data/checkpoints_rac/DAPT_TAPT/TOP_K_3/logs_test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    
    # 新增參數：top_k，建議 default 值設定為與你訓練時相同 (例如 5)
    parser.add_argument("--top_k", type=int, default=3, help="最多使用幾筆鄰居 HTML")
    
    args = parser.parse_args()
    main(args)