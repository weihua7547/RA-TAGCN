import os
import json
import torch
import argparse
import re
from tqdm import tqdm
from preprocessor import html_to_graph_data
import gc

def get_last_chunk_info(output_dir, chunk_size):
    """
    掃描資料夾，回傳最後一個 chunk 的索引以及應該跳過的行數
    """
    if not os.path.exists(output_dir):
        return -1, 0
    
    files = [f for f in os.listdir(output_dir) if f.startswith("chunk_") and f.endswith(".pt")]
    if not files:
        return -1, 0
    
    # 提取所有數字編號
    indices = [int(re.findall(r'\d+', f)[0]) for f in files]
    last_idx = max(indices)
    
    # 假設之前的 chunk 都是滿的 (chunk_size)
    # 跳過行數 = (最大索引 + 1) * chunk_size
    skip_rows = (last_idx + 1) * chunk_size
    return last_idx, skip_rows



def process_jsonl(input_path, output_dir, chunk_size=5000): # 建議縮小 chunk_size
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    last_idx, skip_rows = get_last_chunk_info(output_dir, chunk_size)
    chunk_idx = last_idx + 1
    
    # 這裡不再預先讀取總行數，避免重複 I/O
    print(f"📦 Processing: {input_path}")

    current_chunk = []
    success_count = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=os.path.basename(output_dir))):
            if i < skip_rows:
                continue

            try:
                obj = json.loads(line)
                html = obj.get("html", "")
                graph_data = html_to_graph_data(html)
                
                if graph_data is not None:
                    # 確保轉移到 CPU 並斷開計算圖，減少記憶體占用
                    graph_data.x = graph_data.x.detach().cpu()
                    graph_data.edge_index = graph_data.edge_index.detach().cpu()
                    current_chunk.append(graph_data)
                    success_count += 1
            except Exception:
                continue

            # 達到 chunk 大小就存檔
            if len(current_chunk) >= chunk_size:
                save_path = os.path.join(output_dir, f"chunk_{chunk_idx}.pt")
                torch.save(current_chunk, save_path)
                
                # --- 關鍵優化：釋放記憶體 ---
                del current_chunk
                current_chunk = []
                gc.collect() 
                chunk_idx += 1

    # 處理剩餘的資料
    if current_chunk:
        torch.save(current_chunk, os.path.join(output_dir, f"chunk_{chunk_idx}.pt"))
        del current_chunk
        gc.collect()

    print(f"✅ Finished: {success_count} new samples saved.")

# 其餘 __main__ 部分保持不變

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_in", type=str, default="module1/data_preprocessing/output/pretrain/train.jsonl")
    parser.add_argument("--val_in", type=str, default="module1/data_preprocessing/output/pretrain/val.jsonl")
    parser.add_argument("--test_in", type=str, default="module1/data_preprocessing/output/pretrain/test.jsonl")
    parser.add_argument("--out_root", type=str, default="module1/data_preprocessing/output/pretrain_graph")
    parser.add_argument("--chunk_size", type=int, default=5000)
    args = parser.parse_args()

    # 定義輸出的子資料夾
    tasks = [
        # (args.train_in, os.path.join(args.out_root, "train")),
        # (args.val_in, os.path.join(args.out_root, "val")),
        (args.test_in, os.path.join(args.out_root, "test"))
    ]

    for input_p, output_p in tasks:
        if os.path.exists(input_p):
            process_jsonl(input_p, output_p, args.chunk_size)
        else:
            print(f"⚠️ Skip: {input_p} not found.")