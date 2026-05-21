import os
import json
import time
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn import TAGConv
from torch.utils.data import DataLoader as TorchDataLoader
from preprocessor import html_to_graph_data
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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
# Dataset (single jsonl file, offset-based)
# =========================================================
class HTMLGraphDataset(torch.utils.data.Dataset):
    """
    Each line in jsonl must contain:
      { "html": "...." }
    """
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path
        self.offsets = []
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            offset = 0
            for line in fh:
                self.offsets.append(offset)
                offset += len(line.encode("utf-8"))

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as fh:
                fh.seek(self.offsets[idx])
                obj = json.loads(fh.readline())
                html = obj.get("html", "")
                graph = html_to_graph_data(html)
                if graph is not None:
                    graph.x = graph.x.detach()
                    graph.edge_index = graph.edge_index.detach()
                return graph
        except Exception:
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

def detect_input_dim(jsonl_path):
    """
    Scan jsonl until a valid graph is found, then return node feature dim
    """
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
                graph = html_to_graph_data(obj.get("html", ""))
                if graph is not None:
                    return graph.x.size(1)
            except Exception:
                continue
    raise RuntimeError(f"No valid graph found in {jsonl_path}")

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
    logging.info("🚀 TAGCN Reconstruction Pretraining")
    logging.info(f"Device: {device}")
    logging.info(f"Train jsonl: {args.train_jsonl_path}")
    logging.info(f"Val jsonl: {args.val_jsonl_path}")

    # -------- detect input dim --------
    in_dim = detect_input_dim(args.train_jsonl_path)
    logging.info(f"Detected node feature dim = {in_dim}")

    # -------- model --------
    model = ReconstructionModel(
        in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)

    if args.load_ckpt:
        logging.info(f"🔄 Loading checkpoint: {args.load_ckpt}")
        # 1. 先載入完整的字典
        checkpoint = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        
        # 2. 判斷內容是「純權重」還是「完整字典」
        if 'model_state_dict' in checkpoint:
            # 如果是字典包，提取其中的權重部分
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info("✅ Successfully loaded weights from dict.")
        else:
            # 如果是純權重（舊格式），直接載入
            model.load_state_dict(checkpoint)
            logging.info("✅ Successfully loaded direct weight file.")

    # -------- optimizer & scheduler --------
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # -------- dataset --------
    train_dataset = HTMLGraphDataset(args.train_jsonl_path)
    val_dataset = HTMLGraphDataset(args.val_jsonl_path)
    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")

    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_filter_none
    )
    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_filter_none
    )

    best_val_loss = float("inf")

    # -------- training loop --------
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = validate_epoch(model, val_loader, device)
        logging.info(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.ckpt_dir, "reconstruct_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"✅ Saved best model (val_loss={val_loss:.6f})")

    total_time = time.time() - start_time
    logging.info("🏁 Training finished")
    logging.info(f"⏱ Total time: {total_time / 3600:.2f} hours")
    logging.info(f"📄 Log saved at: {log_path}")

# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl_path", type=str, default="module1/data_preprocessing/output/finetune/train.jsonl")
    parser.add_argument("--val_jsonl_path", type=str, default="module1/data_preprocessing/output/finetune/val.jsonl")
    parser.add_argument("--ckpt_dir", type=str, default="module2/TAGCN/checkpoints_pretrain/DAPT_TAPT")
    parser.add_argument("--load_ckpt", type=str, default="module2/TAGCN/checkpoints_pretrain/DAPT/reconstruct_best.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)        
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)