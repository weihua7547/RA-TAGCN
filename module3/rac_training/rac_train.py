import os
import json
import time
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report
)
from preprocessor import html_to_graph_data
from torch.optim import AdamW
import math
import sys
import warnings
import gc
warnings.filterwarnings("ignore", module='bs4')
warnings.filterwarnings("ignore", module='preprocessor')

sys.stdout.reconfigure(encoding='utf-8')

# =========================================================
# Label map
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
        f"rac_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_path

# =========================================================
# RAC Dataset
# =========================================================
# class RACGraphDataset(torch.utils.data.Dataset):
#     def __init__(self, jsonl_path, label_map=LABEL_MAP, max_neighbors=2):
#         self.label_map = label_map
#         self.max_neighbors = max_neighbors
#         self.file_path = jsonl_path
#         self.offsets = []
        
#         with open(jsonl_path, "r", encoding="utf-8") as f:
#             while True:
#                 offset = f.tell() # 記錄當前指標位置
#                 line = f.readline()
#                 if not line: break
#                 self.offsets.append(offset)

#     def __len__(self):
#         return len(self.offsets)

#     def __getitem__(self, idx):
#         offset = self.offsets[idx]
#         with open(self.file_path, "r", encoding="utf-8") as f:
#             f.seek(offset)
#             obj = json.loads(f.readline())
#         q_graph = html_to_graph_data(obj["query_html"])
#         if q_graph is None or torch.isnan(q_graph.x).any():
#             return None
        
#         # 修改：根據 self.max_neighbors 截取鄰居數量
#         neighbors = []
#         raw_neighbors = obj.get("neighbor_htmls", [])
        
#         # 只取前 K 筆
#         selected_neighbors = raw_neighbors[:self.max_neighbors]
        
#         for h in selected_neighbors:
#             g = html_to_graph_data(h)
#             if g is not None and not torch.isnan(g.x).any():
#                 neighbors.append(g)
        
#         # 如果檢索不到鄰居，使用 Query 本身填充以維持張量形狀
#         if not neighbors:
#             neighbors = [q_graph] * self.max_neighbors
        
#         label = torch.tensor(self.label_map[obj["type"]], dtype=torch.long)
#         return {
#             "query": q_graph,
#             "neighbors": neighbors,
#             "label": label
#         }
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
# TAGCN Encoder (保持不變)
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
# RAC Graph Classifier with Self-Attention (保持不變)
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
# Train / Eval (保持不變)
# =========================================================
def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    running_loss = 0.0
    num_samples = 0
    
    optimizer.zero_grad(set_to_none=True)

    for batch in tqdm(loader, desc=f"Train Epoch {epoch}"):
        if not batch: continue
        
        logits, labels = model(batch, device)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
        
        loss.backward()

        # 每個 Batch 直接更新
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step() # 每步更新

        running_loss += loss.item() * labels.size(0)
        num_samples += labels.size(0)

    return running_loss / num_samples

def evaluate_metrics(model, loader, device, desc="Eval"):
    model.eval()
    total_loss = 0.0
    all_pred, all_gt = [], []
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            if not batch: continue
            logits, labels = model(batch, device)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
            
            total_loss += loss.item() * labels.size(0)
            num_samples += labels.size(0)
            
            pred = logits.argmax(dim=-1)
            all_pred.extend(pred.cpu().tolist())
            all_gt.extend(labels.cpu().tolist())
            
    if num_samples == 0:
        return 0, 0, 0, 0, 0

    avg_loss = total_loss / num_samples
    acc = accuracy_score(all_gt, all_pred)
    # 根據你的要求，一次計算所有指標
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_gt, all_pred, average="macro", zero_division=0
    )
    
    return avg_loss, acc, precision, recall, f1

# =========================================================
# Main
# =========================================================
def finetune(args):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = setup_logger(args.ckpt_dir)
    logging.info(f"Device: {device}")
    logging.info(f"Using TOP_K (max_neighbors): {args.top_k}")

    # 1. 載入 Dataset（加入 Test 集）
    train_ds = RACGraphDataset(args.train_jsonl, max_neighbors=args.top_k)
    val_ds = RACGraphDataset(args.val_jsonl, max_neighbors=args.top_k)
    test_ds = RACGraphDataset(args.test_jsonl, max_neighbors=args.top_k) # 新增

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_rac, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_rac, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_rac, pin_memory=False
    )

    # 2. 偵測 In-dim & 初始化模型
    in_dim = None
    for sample in train_ds:
        if sample and hasattr(sample["query"], 'x') and sample["query"].x is not None:
            in_dim = sample["query"].x.size(1)
            break
    
    encoder = TagcnEncoder(in_dim, args.hidden_dim, args.num_layers)
    model = RACGraphClassifier(encoder, args.hidden_dim, NUM_CLASSES).to(device)

    if args.pretrained_encoder:
        logging.info(f"Loading pretrained encoder: {args.pretrained_encoder}")
        state = torch.load(args.pretrained_encoder, map_location=device)
        model.encoder.load_state_dict(state, strict=False)

    # 3. 優化器與 Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.1)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 4. 訓練迴圈
    best_val_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        # 執行訓練
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        
        # 執行驗證 (Val)
        v_loss, v_acc, v_pre, v_rec, v_f1 = evaluate_metrics(model, val_loader, device, desc=f"Epoch {epoch} [Val]")
        
        # 執行測試 (Test) - 僅供實驗觀察
        t_loss, t_acc, t_pre, t_rec, t_f1 = evaluate_metrics(model, test_loader, device, desc=f"Epoch {epoch} [Test]")

        # 格式化輸出 Log
        logging.info(f"--- Epoch {epoch} Report ---")
        logging.info(f"Train Loss: {train_loss:.4f}")
        logging.info(f"Val  -> Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | Pre: {v_pre:.4f} | Rec: {v_rec:.4f} | F1: {v_f1:.4f}")
        logging.info(f"Test -> Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | Pre: {t_pre:.4f} | Rec: {t_rec:.4f} | F1: {t_f1:.4f}")

        # 僅根據 Val Macro F1 更新最佳模型
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "rac_best.pt"))
            logging.info(f"🏆 Saved best model (Val F1 improved to {v_f1:.4f})")
            
        # 釋放記憶體避免 GNN 累積導致崩潰
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    logging.info(f"Total training time: {total_time / 3600:.2f} hours")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, default="module3/db_data/rac_train_pairs.jsonl")
    parser.add_argument("--val_jsonl", type=str, default="module3/db_data/rac_val_pairs.jsonl")
    parser.add_argument("--test_jsonl", type=str, default="module3/db_data/rac_test_pairs.jsonl")
    parser.add_argument("--ckpt_dir", type=str, default="module3/db_data/checkpoints_rac/DAPT_TAPT/TOP_K_3")
    parser.add_argument("--pretrained_encoder", type=str, default="module2/TAGCN/checkpoints_pretrain/DAPT_TAPT/reconstruct_best.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    
    # 新增參數：TOP_K
    parser.add_argument("--top_k", type=int, default=3, help="最多使用幾筆鄰居 HTML")
    
    args = parser.parse_args()
    finetune(args)