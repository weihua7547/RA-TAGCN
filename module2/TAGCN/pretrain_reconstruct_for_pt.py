import os
import json
import time
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import torch
import re
import torch.nn.functional as F
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn import TAGConv
from torch.utils.data import DataLoader as TorchDataLoader
from preprocessor import html_to_graph_data
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import glob
from torch_geometric.data.data import DataEdgeAttr
from torch_geometric.data import Data
import random
import math

# 允許加載 PyG 的特定類別
torch.serialization.add_safe_globals([Data, DataEdgeAttr])

# =========================================================
# Logging
# =========================================================
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
# 🆕 Checkpoint Helper
# =========================================================
def find_latest_checkpoint(ckpt_dir):
    """ 在目錄中尋找 epoch 最大的 pt 檔案 """
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "reconstruct_epoch_*.pt"))
    if not ckpt_files:
        return None
    
    # 提取檔名中的數字並排序
    def extract_epoch(filename):
        match = re.search(r'epoch_(\d+)\.pt', filename)
        return int(match.group(1)) if match else -1
    
    latest_ckpt = max(ckpt_files, key=extract_epoch)
    return latest_ckpt

# =========================================================
# Dataset (single jsonl file, offset-based)
# =========================================================
class ShardedGraphDataset(torch.utils.data.IterableDataset):
    def __init__(self, folder_path, shuffle=True):
        super().__init__()
        self.pt_files = sorted(glob.glob(os.path.join(folder_path, "*.pt")))
        self.shuffle = shuffle
        
    def __iter__(self):
        # 複製一份檔案清單
        file_list = self.pt_files[:]
        if self.shuffle:
            random.shuffle(file_list)
        
        # 多進程處理：讓每個 Worker 只分到一部份檔案，避免重複讀取
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # 根據 worker id 切分檔案清單
            per_worker = int(math.ceil(len(file_list) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(file_list))
            file_list = file_list[iter_start:iter_end]

        for file_path in file_list:
            # 這裡就是關鍵：一次開啟，直接噴出 5000 筆資料才換下一個
            data_list = torch.load(file_path, weights_only=False)
            if self.shuffle:
                random.shuffle(data_list)
            
            for data in data_list:
                yield data

    def __len__(self):
        return self.total_count

    def __getitem__(self, idx):
        # 計算 idx 落在哪個檔案
        file_idx = idx // self.chunk_size
        inner_idx = idx % self.chunk_size
        
        # 如果請求的檔案不是目前載入的檔案，則更換
        if file_idx != self.current_file_idx:
            if file_idx >= len(self.pt_files):
                return None
            self.data_list = torch.load(self.pt_files[file_idx], weights_only=False)
            self.current_file_idx = file_idx
        
        if inner_idx < len(self.data_list):
            return self.data_list[inner_idx]
        return None

# =========================================================
# Model
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

class FeatureDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z):
        return self.mlp(z)

class ReconstructionModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.encoder = TagcnEncoder(in_dim, hidden_dim, num_layers)
        self.decoder = FeatureDecoder(hidden_dim, in_dim)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_rec = self.decoder(z)
        return x_rec, z

# =========================================================
# Helpers
# =========================================================
def collate_filter_none(batch):
    return [d for d in batch if d is not None]

def batch_to_pyg(batch_list, device):
    if batch_list is None or len(batch_list) == 0:
        return None
    return PyGBatch.from_data_list(batch_list).to(device)

def detect_input_dim(folder_path):
    pt_files = glob.glob(os.path.join(folder_path, "*.pt"))
    if not pt_files:
        raise RuntimeError("No .pt files found!")
    # 加入 weights_only=False
    data_list = torch.load(pt_files[0], weights_only=False)
    return data_list[0].x.size(1)

# =========================================================
# Train / Validate
# =========================================================
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, n_nodes = 0.0, 0
    for batch_list in tqdm(loader, desc="Training", leave=False):
        batch = batch_to_pyg(batch_list, device)
        if batch is None:
            continue
        optimizer.zero_grad()
        x_rec, _ = model(batch.x, batch.edge_index)
        loss = F.mse_loss(x_rec, batch.x)
        if torch.isnan(loss):
            logging.warning("NaN loss detected, skipping batch")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防梯度爆炸
        optimizer.step()
        scheduler.step()  # CosineAnnealingLR 每 step 更新
        total_loss += loss.item() * batch.x.size(0)
        n_nodes += batch.x.size(0)
    return total_loss / max(1, n_nodes)

def validate_epoch(model, loader, device):
    model.eval()
    total_loss, n_nodes = 0.0, 0
    with torch.no_grad():
        for batch_list in tqdm(loader, desc="Validation", leave=False):
            batch = batch_to_pyg(batch_list, device)
            if batch is None:
                continue
            x_rec, _ = model(batch.x, batch.edge_index)
            loss = F.mse_loss(x_rec, batch.x)
            total_loss += loss.item() * batch.x.size(0)
            n_nodes += batch.x.size(0)
    return total_loss / max(1, n_nodes)

# =========================================================
# Main
# =========================================================
def train(args):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = setup_logger(args.ckpt_dir)
    logging.info("🚀 TAGCN Reconstruction Pretraining (with Checkpoint Resume)")

    # -------- detect input dim --------
    in_dim = detect_input_dim(args.train_pt_dir)
    logging.info(f"Detected node feature dim = {in_dim}")

    # -------- model --------
    model = ReconstructionModel(
        in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)

    # -------- optimizer & scheduler --------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # -------- 🆕 斷點續傳機制 --------
    start_epoch = 1
    best_val_loss = float("inf")

    # 優先檢查手動指定的 load_ckpt，否則自動搜尋目錄
    checkpoint_path = args.load_ckpt if args.load_ckpt else find_latest_checkpoint(args.ckpt_dir)

    if checkpoint_path and os.path.exists(checkpoint_path):
        logging.info(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 兼容只儲存模型與儲存完整狀態兩種格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float("inf"))
            logging.info(f"⏩ Resuming from Epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            logging.info("⚠️ Old checkpoint format detected (weights only)")

    # -------- dataset --------
    # -------- dataset (in main train function) --------
    # 注意：args 需要增加對應的資料夾路徑參數  
    train_dataset = ShardedGraphDataset(args.train_pt_dir,shuffle=True)
    val_dataset = ShardedGraphDataset(args.val_pt_dir,shuffle=True)

    # 關鍵：num_workers 不要設太高 (例如 2 或 4)，
    # 因為每個 worker 都會嘗試載入一個 5000 筆的 .pt 檔，設太高會爆記憶體。
    train_loader = TorchDataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        # shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=collate_filter_none
    )
    val_loader = TorchDataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        # shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=collate_filter_none
    )

    # -------- training loop --------
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = validate_epoch(model, val_loader, device)
        logging.info(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # 儲存狀態字典
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }

        # 💾 保存最新一個 Epoch (斷點續傳用)
        latest_ckpt_path = os.path.join(args.ckpt_dir, f"reconstruct_epoch_{epoch}.pt")
        torch.save(state, latest_ckpt_path)
        
        # 刪除上一個 Epoch 的舊檔案，避免硬碟爆掉（只保留最後一個和最好的）
        prev_ckpt = os.path.join(args.ckpt_dir, f"reconstruct_epoch_{epoch-1}.pt")
        if os.path.exists(prev_ckpt):
            os.remove(prev_ckpt)

        # 🏆 保存 Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state['best_val_loss'] = best_val_loss
            best_ckpt_path = os.path.join(args.ckpt_dir, "reconstruct_best.pt")
            torch.save(state, best_ckpt_path)
            logging.info(f"✅ Saved best model (val_loss={val_loss:.6f})")

    total_time = time.time() - start_time
    logging.info(f"🏁 Training finished. Total time: {total_time / 3600:.2f} hours")

# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pt_dir", type=str, default="module1/data_preprocessing/output/pretrain_graph/train/")
    parser.add_argument("--val_pt_dir", type=str, default="module1/data_preprocessing/output/pretrain_graph/val/")
    parser.add_argument("--ckpt_dir", type=str, default="module2/TAGCN/checkpoints_pretrain/DAPT")
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)        
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)