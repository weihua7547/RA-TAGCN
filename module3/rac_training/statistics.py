import json
from collections import Counter
from tqdm import tqdm

def analyze_neighbor_stats(file_path):
    counts = []
    total_records = 0

    print(f"📊 正在讀取並分析: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="掃描進度"):
                try:
                    obj = json.loads(line)
                    # 取得 neighbor_htmls 的數量，若不存在則視為 0
                    n_count = len(obj.get("neighbor_htmls", []))
                    counts.append(n_count)
                    total_records += 1
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到檔案 {file_path}")
        return

    if total_records == 0:
        print("⚠️ 檔案中沒有資料。")
        return

    # 使用 Counter 統計出現次數
    stats = Counter(counts)
    
    print("\n" + "="*45)
    print(f"{'鄰居數量':^10} | {'檔案筆數':^10} | {'百分比 (%)':^15}")
    print("-" * 45)

    # 按照數量從 5 到 0 排序顯示
    for i in range(5, -1, -1):
        num_records = stats.get(i, 0)
        percentage = (num_records / total_records) * 100
        # 使用直條圖簡單視覺化 (Sparkline 概念)
        bar = "█" * int(percentage / 5) 
        print(f"{i:^12} | {num_records:^12} | {percentage:>12.2f}%  {bar}")

    print("-" * 45)
    print(f"{'總計':^12} | {total_records:^12} | {'100.00%':^15}")
    print("="*45)

if __name__ == "__main__":
    # 設定你剛剛產出的檔案路徑
    OUTPUT_PATH = "module3/db_data/rac_train_pairs.jsonl"
    analyze_neighbor_stats(OUTPUT_PATH)
#train
# =============================================
#    鄰居數量    |    檔案筆數    |     百分比 (%)    
# ---------------------------------------------
#      5       |    75854     |        96.97%  
#      4       |     268      |         0.34%  
#      3       |     291      |         0.37%  
#      2       |     377      |         0.48%  
#      1       |     511      |         0.65%  
#      0       |     921      |         1.18%  
# ---------------------------------------------
#      總計      |    78222     |     100.00%    
# =============================================
#val 
# =============================================
#    鄰居數量    |    檔案筆數    |     百分比 (%)    
# ---------------------------------------------
#      5       |     6283     |        96.37%  
#      4       |      81      |         1.24%  
#      3       |      18      |         0.28%  
#      2       |      40      |         0.61%  
#      1       |      40      |         0.61%  
#      0       |      58      |         0.89%  
# ---------------------------------------------
#      總計      |     6520     |     100.00%    
# =============================================
#test
# =============================================
#    鄰居數量    |    檔案筆數    |     百分比 (%)    
# ---------------------------------------------
#      5       |     6918     |        98.24%  
#      4       |      17      |         0.24%  
#      3       |      9       |         0.13%  
#      2       |      17      |         0.24%  
#      1       |      28      |         0.40%  
#      0       |      53      |         0.75%  
# ---------------------------------------------
#      總計      |     7042     |     100.00%    
# =============================================