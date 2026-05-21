import os
import json
import argparse
import logging
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn import TAGConv, global_mean_pool

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

from preprocessor import html_to_graph_data
import gc

# =========================================================
# Fixed label map (與訓練保持一致)
# =========================================================
LABEL_MAP = {
    "benign": 0,
    "phishing": 1,
    "defacement": 2,
    "malware": 3,
}


# =========================================================
# Logging
# =========================================================
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
# Dataset
# =========================================================
class LabeledHTMLGraphDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, label_map=LABEL_MAP):
        self.jsonl_path = jsonl_path
        self.offsets = []
        self.labels = []
        self.label_map = label_map

        with open(jsonl_path, "rb") as fh:
            offset = 0
            for line in fh:
                try:
                    obj = json.loads(line.decode("utf-8"))
                    lbl = obj.get("label") or obj.get("type")

                    # 只保留存在於label_map的樣本
                    if lbl in self.label_map:
                        self.offsets.append(offset)
                        self.labels.append(lbl)

                except Exception:
                    pass

                offset = fh.tell()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        try:
            with open(self.jsonl_path, "rb") as fh:
                fh.seek(self.offsets[idx])
                obj = json.loads(fh.readline().decode("utf-8"))

                graph = html_to_graph_data(obj.get("html", ""))
                if graph is None:
                    return None

                graph.x = graph.x.detach()
                graph.edge_index = graph.edge_index.detach()

                graph.y = torch.tensor(
                    self.label_map[self.labels[idx]],
                    dtype=torch.long
                )
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
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
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
# Helpers
# =========================================================
def collate_filter_none(batch):
    return [b for b in batch if b is not None]


def batch_to_pyg(batch_list, device):
    if batch_list is None or len(batch_list) == 0:
        return None
    return PyGBatch.from_data_list(batch_list).to(device)


# =========================================================
# Test
# =========================================================
def test(model, loader, device):
    model.eval()
    all_pred, all_gt = [], []

    with torch.no_grad():
        for batch_list in tqdm(loader, desc="Testing"):
            batch = batch_to_pyg(batch_list, device)
            if batch is None:
                continue

            logits = model(batch)
            pred = logits.argmax(dim=-1)

            all_pred.extend(pred.cpu().tolist())
            all_gt.extend(batch.y.cpu().tolist())

    acc = accuracy_score(all_gt, all_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_gt,
        all_pred,
        average="macro",
        zero_division=0
    )

    logging.info("===== TEST RESULT =====")
    logging.info(f"Accuracy : {acc:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall   : {recall:.4f}")
    logging.info(f"F1-score : {f1:.4f}")

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
# Main
# =========================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_logger(args.log_dir)

    logging.info(f"Using fixed label map: {LABEL_MAP}")

    dataset = LabeledHTMLGraphDataset(args.test_jsonl_path)

    loader = TorchDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_filter_none
    )

    # detect input dim
    in_dim = None
    for i in range(len(dataset)):
        g = dataset[i]
        if g is not None:
            in_dim = g.x.size(1)
            break

    if in_dim is None:
        raise RuntimeError("Cannot detect node feature dim")

    encoder = TagcnEncoder(
        in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )

    model = GraphClassifier(
        encoder,
        hidden_dim=args.hidden_dim,
        num_classes=len(LABEL_MAP)
    ).to(device)

    logging.info(f"Loading model: {args.model_path}")
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    test(model, loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_jsonl_path", type=str, default="module1/data_preprocessing/output/finetune_no_leak/test.jsonl")
    parser.add_argument("--model_path", type=str, default="module2/TAGCN/checkpoints_finetune/DAPT+TAPT/finetune_best.pt")
    parser.add_argument("--log_dir", type=str, default="module2/TAGCN/checkpoints_finetune/DAPT+TAPT/logs_test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)

    args = parser.parse_args()
    main(args)
