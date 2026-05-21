import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch as PyGBatch
from tqdm import tqdm
import time
from preprocessor import html_to_graph_data
from pretrain_reconstruct import (
    HTMLGraphDataset,
    ReconstructionModel,
    collate_filter_none,
    batch_to_pyg,
    detect_input_dim
)

# =========================================================
# Test RMSE
# =========================================================
def test_rmse(model, loader, device):
    start_time = time.time()
    model.eval()
    total_squared_error = 0.0
    total_nodes = 0

    with torch.no_grad():
        for batch_list in tqdm(loader, desc="Testing"):
            batch = batch_to_pyg(batch_list, device)
            if batch is None:
                continue

            x_rec, _ = model(batch.x, batch.edge_index)

            # sum squared error
            squared_error = F.mse_loss(
                x_rec, batch.x, reduction="sum"
            )

            total_squared_error += squared_error.item()
            total_nodes += batch.x.numel()  # total feature elements

    mse = total_squared_error / total_nodes
    rmse = mse ** 0.5
    total_time = time.time() - start_time
    print(f"⏱ Total time: {total_time / 60:.2f} minutes")
    print("\n===== Test Result =====")
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")

    return rmse


# =========================================================
# Main
# =========================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    # detect input dim
    in_dim = detect_input_dim(args.test_jsonl_path)
    print("Detected feature dim:", in_dim)

    # load model
    model = ReconstructionModel(
        in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)

    if args.ckpt_path and os.path.exists(args.ckpt_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("⚠️ No checkpoint loaded, using random initialized weights.")

    # dataset
    test_dataset = HTMLGraphDataset(args.test_jsonl_path)

    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_filter_none
    )

    test_rmse(model, test_loader, device)


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_jsonl_path",
        type=str,
        default="module1/data_preprocessing/output/finetune/test.jsonl"
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None#"module2/TAGCN/checkpoints_pretrain/DAPT_TAPT/reconstruct_best.pt"
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)

    args = parser.parse_args()
    main(args)
