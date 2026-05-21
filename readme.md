RA-TAGCN是一個整合檢索強化(Retrieval Augmentation)與拓樸自適應圖卷積網路(Topology Adaptive Graph Convolution Network, TAGCN)的模型框架。
結合檢索強化的優點，可以在向量資料庫中搜尋近似的歷史樣本，提供模型更多的線索去判斷。並且當有新形態的樣本出現時，可以新增至向量資料庫中獲得部分的知識而非只能重頭訓練。
# 注意
此儲存庫僅展示相關程式碼範本與模型範本。資料集部分檔案過大約70GB，尚未上傳。
# Framework
以惡意網頁分類為例(良性、釣魚、竄改、惡意程式散播)。
向量資料庫是透過bge-small-en-v1.5將html轉換為固定384維的語意向量。並使用FAISS做為向量庫底層查詢方式，透過ANN查詢出50筆的候選池之後使用COS餘弦相似度進行排序並排除掉低於0.65相似度分數的樣本。透過這種方式，連接到語意模態。
![RA-TAGCN框架](/picture/RAC-TAGCN框架.png)
![融合分類層架構](/picture/RAC-TAGCN融合分類層.png)
# 圖轉換
HTML的文件物件模型(Document Object Model, DOM)可以視作一顆樹，樹及為圖的一種表示方式，所以可以被GNN處理。
其中標籤視作節點且父子關係則為有方向的邊。
![DOM Tree示意圖](/picture/網頁圖結構示意圖.png)
接下來，只有圖的形狀是不太能夠被GNN直接處理的，還得需嵌入節點的特徵。
這裡透過查表方式嵌入特徵，如下圖所示，共會取得節點+標籤+其他特徵共23維。
![標籤嵌入查表](/picture/標籤嵌入查表.png)
![屬性嵌入查表](/picture/屬性嵌入查表.png)
![節點特徵拼接](/picture/節點特徵拼接.png)
# Result
由於實驗為多元分類實驗，F1分數皆採取Macro-F1，使實驗結果更關注少數樣本。
## 檢索筆數分析
透過Top-K筆的鄰居樣本進行實驗，比較K在1~5的情況下，何者可以取得較高的F1成績。由圖中
![RA-TAGCN結果](/result/實驗結果RA-TAGCN.png)
## 消融實驗
將比較有沒有使用檢索模組的好壞。由圖中，可以看到具有檢索模組的情況下，F1分數具有明顯的進步，顯示檢索模組能幫助少數樣本類別的辨識。
![RA-TAGCNvsMHTML-TAGCN](/result/實驗結果RA-TAGCNvsMHTML-TAGCN.png)
