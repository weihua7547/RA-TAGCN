import os
import json
import time
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn import TAGConv, global_mean_pool
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from preprocessor import html_to_graph_data
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc
# =========================================================
# Fixed label map
# =========================================================
LABEL_MAP = {
    "benign": 0,
    "phishing": 1,
    "defacement": 2,
    "malware": 3,
}

# =========================================================
# Logging setup
# =========================================================
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
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
# Dataset: Eager Caching (初始化時完成轉換)
# =========================================================
class LabeledHTMLGraphDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, label_map=LABEL_MAP):
        self.label_map = label_map
        self.cache = [] 
        
        file_size = os.path.getsize(jsonl_path)
        logging.info(f"Loading {jsonl_path} ({file_size / 1024**2:.2f} MB)...")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            # 使用 unit='B', unit_scale=True 來顯示讀取的位元組進度
            pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Processing {os.path.basename(jsonl_path)}")
            
            count = 0
            for line in f:
                # 更新進度條：增加該行字串轉成 bytes 後的長度
                pbar.update(len(line.encode('utf-8')))
                
                try:
                    obj = json.loads(line)
                    lbl = obj.get("label") or obj.get("type")
                    if lbl not in self.label_map:
                        continue
                    
                    graph = html_to_graph_data(obj.get("html", ""))
                    if graph is not None:
                        # 記憶體優化：確保轉向 CPU 並脫離計算圖
                        graph.x = graph.x.detach().cpu()
                        graph.edge_index = graph.edge_index.detach().cpu()
                        graph.y = torch.tensor(self.label_map[lbl], dtype=torch.long)
                        self.cache.append(graph)
                        count += 1
                except Exception:
                    continue
                
                # 每 5000 筆執行一次垃圾回收，清除 BeautifulSoup 或 JSON 解析產生的碎物
                if count > 0 and count % 5000 == 0:
                    gc.collect()
            
            pbar.close()
            
        logging.info(f"Successfully cached {len(self.cache)} samples.")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]

# =========================================================
# Model Architecture
# =========================================================
class TagcnEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropout = dropout
        self.convs.append(TAGConv(in_dim, hidden_dim))
        self.norms.append(torch.nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(TAGConv(hidden_dim, hidden_dim))
            self.norms.append(torch.nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GraphClassifier(torch.nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.pool = global_mean_pool
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, batch):
        z = self.encoder(batch.x, batch.edge_index)
        g = self.pool(z, batch.batch)
        return self.classifier(g)

# =========================================================
# Training & Evaluation Functions
# =========================================================
def collate_filter_none(batch):
    return [b for b in batch if b is not None]

def batch_to_pyg(batch_list, device):
    if not batch_list: return None
    return PyGBatch.from_data_list(batch_list).to(device)

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, total_graphs = 0.0, 0
    for batch_list in loader:
        batch = batch_to_pyg(batch_list, device)
        if batch is None: continue
        
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y, label_smoothing=0.1)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / max(1, total_graphs)

def eval_epoch(model, loader, device):
    model.eval()
    all_pred, all_gt = [], []
    total_loss, total_graphs = 0.0, 0
    
    with torch.no_grad():
        for batch_list in loader:
            batch = batch_to_pyg(batch_list, device)
            if batch is None: continue
            
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y)
            
            pred = logits.argmax(dim=-1)
            all_pred.extend(pred.cpu().tolist())
            all_gt.extend(batch.y.cpu().tolist())
            
            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs
            
    if not all_gt: return 0.0, 0.0, 0.0, 0.0, 0.0
    
    avg_loss = total_loss / max(1, total_graphs)
    acc = accuracy_score(all_gt, all_pred)
    pre = precision_score(all_gt, all_pred, average="macro", zero_division=0)
    rec = recall_score(all_gt, all_pred, average="macro", zero_division=0)
    f1 = f1_score(all_gt, all_pred, average="macro", zero_division=0)
    
    return avg_loss, acc, pre, rec, f1

# =========================================================
# Main Finetune Process
# =========================================================
def finetune(args):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = setup_logger(args.ckpt_dir)
    
    logging.info(f"🚀 Device: {device} | Experiment: {args.ckpt_dir}")
    
    # Dataset Loading (with Eager Cache)
    train_ds = LabeledHTMLGraphDataset(args.train_jsonl_path)
    val_ds   = LabeledHTMLGraphDataset(args.val_jsonl_path)
    test_ds  = LabeledHTMLGraphDataset(args.test_jsonl_path)
    
    # Detect in_dim
    in_dim = train_ds[0].x.size(1) if len(train_ds) > 0 else 64
    
    # Model & Optimizer
    encoder = TagcnEncoder(in_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    model = GraphClassifier(encoder, hidden_dim=args.hidden_dim, num_classes=len(LABEL_MAP)).to(device)
    
    if args.pretrained_encoder and os.path.exists(args.pretrained_encoder):
        logging.info(f"Loading pretrained encoder: {args.pretrained_encoder}")
        model.encoder.load_state_dict(torch.load(args.pretrained_encoder, map_location=device), strict=False)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    train_loader = TorchDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_filter_none)
    val_loader   = TorchDataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_filter_none)
    test_loader  = TorchDataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_filter_none)
    
    best_val_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        # Eval Val
        v_loss, v_acc, v_pre, v_rec, v_f1 = eval_epoch(model, val_loader, device)
        # Eval Test
        t_loss, t_acc, t_pre, t_rec, t_f1 = eval_epoch(model, test_loader, device)
        
        # Log requirements
        logging.info(f"[Epoch {epoch:03}] Train Loss: {train_loss:.4f}")
        logging.info(f"[Epoch {epoch:03}] Val  -> Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | Pre: {v_pre:.4f} | Rec: {v_rec:.4f} | F1: {v_f1:.4f}")
        logging.info(f"[Epoch {epoch:03}] Test -> Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | Pre: {t_pre:.4f} | Rec: {t_rec:.4f} | F1: {t_f1:.4f}")
        
        # Update best model based on Val F1
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "finetune_best.pt"))
            logging.info(f"⭐ New Best Val F1: {v_f1:.4f} (Model Saved)")
            
    logging.info(f"🏁 Finished. Total Time: {(time.time()-start_time)/3600:.2f}h")

def run_finetune(base_args, ckpt_dir, pretrained_encoder):
    args = argparse.Namespace(**vars(base_args))
    args.ckpt_dir = ckpt_dir
    args.pretrained_encoder = pretrained_encoder
    finetune(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl_path", type=str, default="module1/data_preprocessing/output/finetune/train.jsonl")
    parser.add_argument("--val_jsonl_path",   type=str, default="module1/data_preprocessing/output/finetune/val.jsonl")
    parser.add_argument("--test_jsonl_path",  type=str, default="module1/data_preprocessing/output/finetune/test.jsonl")
    parser.add_argument("--batch_size",       type=int, default=32)
    parser.add_argument("--hidden_dim",       type=int, default=64)
    parser.add_argument("--epochs",           type=int, default=50)
    parser.add_argument("--num_layers",       type=int, default=3)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--pretrained_encoder", type=str, default=None) 
    common_args = parser.parse_args()

    experiments = [
        # ("module2/TAGCN/checkpoints_finetune/DAPT/", "module2/TAGCN/checkpoints_pretrain/TAPT/reconstruct_best.pt"),
        ("module2/TAGCN/checkpoints_finetune/DAPT_TAPT/", "module2/TAGCN/checkpoints_pretrain/DAPT_TAPT/reconstruct_best.pt"),
        {"module2/TAGCN/checkpoints_finetune/None/", None}
    ]

    for ckpt, pretrain in experiments:
        run_finetune(common_args, ckpt, pretrain)