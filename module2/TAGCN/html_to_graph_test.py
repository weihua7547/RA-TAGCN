# html_to_graph_test.py
from preprocessor import html_to_graph_data_label
import networkx as nx
import matplotlib.pyplot as plt

html_path = "raw.html"  # ← 直接改這裡

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

graph = html_to_graph_data_label(html)

if graph is None:
    print("❌ HTML 無法轉成 graph")
    exit()

print("✅ HTML successfully converted to graph")
print(f"Number of nodes: {graph.x.size(0)}")
print(f"Node feature dim: {graph.x.size(1)}")
print(f"Number of edges: {graph.edge_index.size(1)}")

# ========= PyG → NetworkX =========
G = nx.DiGraph()  # DOM 是有方向的（parent -> child）

edge_index = graph.edge_index.cpu().numpy()
num_nodes = graph.x.size(0)

G.add_nodes_from(range(num_nodes))
for src, dst in edge_index.T:
    G.add_edge(int(src), int(dst))
labels = {i: graph.node_labels[i] for i in range(len(graph.node_labels))}
# ========= Draw =========
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.15, seed=42)  # 自動排版
nx.draw(
    G,
    pos,
    node_size=30,
    node_color="lightblue",
    edge_color="gray",
        labels=labels,       # ← 加上這行
    font_size=8,         # 字體大小可調整
)

plt.title("HTML DOM Graph Visualization")
plt.show()
