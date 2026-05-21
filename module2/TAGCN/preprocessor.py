import os
import json
from tqdm import tqdm
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import torch
from torch_geometric.data import Data
import gc
import warnings
import traceback
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ========== 詞彙表 ==========
TAG_VOCAB = [
    "html", "head", "body", "div", "span", "a", "img", "script", "iframe",
    "form", "input", "meta", "link", "button", "p", "h1", "h2", "ul", "li"
]

ATTR_VOCAB = [
    "href", "src", "onclick", "onload", "action", "method", "id", "class",
    "style", "name", "value", "type", "alt", "title", "content"
]
# ========== 預先定義好的對應表 (放全域加速查詢) ==========
TAG_TO_ID = {tag: i + 1 for i, tag in enumerate(TAG_VOCAB)}
ATTR_TO_ID = {attr: i + 1 for i, attr in enumerate(ATTR_VOCAB)}

TAG_EMB_DIM = 8
ATTR_EMB_DIM = 8

tag_embedding = torch.nn.Embedding(len(TAG_VOCAB) + 1, TAG_EMB_DIM)
attr_embedding = torch.nn.Embedding(len(ATTR_VOCAB) + 1, ATTR_EMB_DIM)

def _shannon_entropy(s: str) -> float:
    from math import log2
    if not s:
        return 0.0
    freq = {c: s.count(c) / len(s) for c in set(s)}
    return -sum(p * log2(p) for p in freq.values())

def safe_parse_html(html):
    for parser in ["lxml", "html.parser", "html5lib"]:
        try:
            return BeautifulSoup(html, parser)
        except Exception:
            continue
    return None

def parse_html_to_graph(soup):
    nodes, edges = [], []
    stack = [(soup, None)]
    while stack:
        node, parent_idx = stack.pop()
        if not hasattr(node, "name") or node.name is None:
            continue
        idx = len(nodes)
        nodes.append(node)
        if parent_idx is not None:
            edges.append((parent_idx, idx))
        for child in getattr(node, "children", []):
            stack.append((child, idx))
    return nodes, edges

def extract_node_features(node, tag_to_id, attr_to_id):
    # 1. 避免使用 list(node.parents) 這種 O(depth) 操作
    # 如果只是要深度，可以在 parse 過程中的 stack 順便傳遞 depth
    
    tag_name = node.name or ""
    tag_id = tag_to_id.get(tag_name, 0)

    attrs = node.attrs or {}
    attr_ids = [attr_to_id.get(k, 0) for k in attrs.keys() if k in ATTR_VOCAB]
    
    # 用字串包含判斷，避免重複 join
    attr_text = str(attrs) 
    has_url = 1 if "[URL]" in attr_text else 0
    has_base64 = 1 if "[BASE64]" in attr_text else 0
    has_hash = 1 if "[HASH]" in attr_text else 0
    has_num = 1 if "[NUM]" in attr_text else 0

    # 回傳純 list，最後再一次轉 Tensor
    return [tag_id, attr_ids, has_url, has_base64, has_hash, has_num]


def html_to_graph_data(html: str) -> Data:
    """
    高度優化版：
    1. 使用 lxml 直接解析
    2. 單次遍歷 (O(N)) 同時提取結構與特徵
    3. 避免在迴圈內建立 Tensor
    """
    if not html:
        return None
        
    # 直接指定 lxml，這是目前最快的 parser
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as e:
        print("Parse error:", e)
        traceback.print_exc()
        return None

    root = soup.body or soup.html or soup
    
    nodes_features = []
    edges = []
    
    # stack 儲存: (節點物件, 父節點索引, 當前深度)
    stack = [(root, None, 0)]
    
    while stack:
        node, parent_idx, depth = stack.pop()
        
        # 過濾掉非標籤節點 (如純文字)
        if not hasattr(node, "name") or node.name is None:
            continue
            
        current_idx = len(nodes_features)
        
        # --- 1. 提取標籤與屬性特徵 (純 Python 運算) ---
        tag_id = TAG_TO_ID.get(node.name, 0)
        
        attrs = node.attrs or {}
        attr_ids = [ATTR_TO_ID.get(k, 0) for k in attrs.keys() if k in ATTR_TO_ID]
        
        # 結構特徵
        children = [c for c in node.children if hasattr(c, "name") and c.name is not None]
        num_children = len(children)
        is_leaf = 1.0 if num_children == 0 else 0.0
        
        # 關鍵內容檢查 (避免 join，直接轉字串判斷)
        attr_str = str(attrs)
        has_url = 1.0 if "[URL]" in attr_str else 0.0
        has_base64 = 1.0 if "[BASE64]" in attr_str else 0.0
        has_hash = 1.0 if "[HASH]" in attr_str else 0.0
        has_num = 1.0 if "[NUM]" in attr_str else 0.0

        # 將特徵存為 list，稍後一次性轉換
        nodes_features.append({
            "tag_id": tag_id,
            "attr_ids": attr_ids,
            "struct": [float(depth), float(num_children), is_leaf, has_url, has_base64, has_hash, has_num]
        })
        
        # --- 2. 建立邊 ---
        if parent_idx is not None:
            edges.append((parent_idx, current_idx))
            
        # --- 3. 繼續走訪子節點 ---
        for child in reversed(children):
            stack.append((child, current_idx, depth + 1))

    if not nodes_features:
        return None

    # --- 4. 批次轉換為 Tensor (效能關鍵) ---
    with torch.no_grad():
        # 批次處理 Tag Embedding
        tag_ids_t = torch.tensor([n["tag_id"] for n in nodes_features], dtype=torch.long)
        tag_embs = tag_embedding(tag_ids_t) # [N, 8]
        
        # 批次處理 Attr Embedding (取平均)
        attr_embs_list = []
        for n in nodes_features:
            if n["attr_ids"]:
                a_ids = torch.tensor(n["attr_ids"], dtype=torch.long)
                attr_embs_list.append(attr_embedding(a_ids).mean(dim=0))
            else:
                attr_embs_list.append(torch.zeros(ATTR_EMB_DIM))
        attr_embs = torch.stack(attr_embs_list) # [N, 8]
        
        # 結構特徵
        struct_feats = torch.tensor([n["struct"] for n in nodes_features], dtype=torch.float) # [N, 7]
        
        # 最終合併所有特徵：8 + 8 + 7 = 23 維
        x = torch.cat([tag_embs, attr_embs, struct_feats], dim=-1)

    # 建立 Edge Index
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)
def html_to_graph_data_label(html: str) -> Data:
    soup = safe_parse_html(html)
    if soup is None:
        return None
    nodes, edges = parse_html_to_graph(soup)

    node_features = []
    node_labels = []  # ← 新增，存 tag 名稱
    for n in nodes:
        node_features.append(extract_node_features(n))
        node_labels.append(n.name or "")  # ← 存 tag 名稱

    if not node_features:
        return None

    x = torch.stack(node_features)
    edge_index = torch.tensor(edges).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.node_labels = node_labels  # ← 把標籤存進 Data

    return data

def test_all_html(json_dir):
    total_ok = 0
    total_fail = 0
    total_all = 0

    for file in os.listdir(json_dir):
        if not file.endswith(".jsonl"):
            continue

        path = os.path.join(json_dir, file)

        # 先算行數，讓 tqdm 有 total
        with open(path, "r", encoding="utf-8") as f:
            num_lines = sum(1 for _ in f)

        with open(path, "r", encoding="utf-8") as f:
            pbar = tqdm(
                f,
                total=num_lines,
                desc=f"Parsing {file}",
                leave=False
            )

            for line in pbar:
                total_all += 1
                try:
                    obj = json.loads(line)
                    html = obj.get("html", "")
                    data = html_to_graph_data(html)

                    if data is None:
                        total_fail += 1
                    else:
                        total_ok += 1

                except Exception:
                    total_fail += 1

                finally:
                    if "data" in locals():
                        del data
                    if "html" in locals():
                        del html
                    if "obj" in locals():
                        del obj
                    gc.collect()

                pbar.set_postfix({
                    "ok": total_ok,
                    "fail": total_fail
                })

    print("========== SUMMARY ==========")
    print(f"Total samples: {total_all}")
    print(f"Valid graphs:  {total_ok}")
    print(f"Failed:        {total_fail}")
    print(f"Success rate:  {total_ok / max(1, total_all):.4f}")


if __name__ == "__main__":
    # test_all_html("../../html_data/crawl_common/")
    test_all_html("module1/data_preprocessing/output/finetune")
