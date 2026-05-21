from torch_geometric.data import Batch
import torch
def rac_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    q_graphs, r1_graphs, r2_graphs, labels = zip(*batch)

    return (
        Batch.from_data_list(q_graphs),
        Batch.from_data_list(r1_graphs),
        Batch.from_data_list(r2_graphs),
        torch.stack(labels)
    )
