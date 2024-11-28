import torch
from torch_geometric.nn import GPSConv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures

# Network architecture 
class GPSNetwork(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.2):
        super(GPSNetwork, self).__init__()
        self.gps1 = GPSConv(
            channels=hidden_channels,
            conv=None,  # Use default message-passing scheme
            heads=heads,
            dropout=dropout,
            act="relu",
            norm="batch_norm",
            attn_type="multihead",
        )
        self.gps2 = GPSConv(
            channels=hidden_channels,
            conv=None,
            heads=heads,
            dropout=dropout,
            act="relu",
            norm="batch_norm",
            attn_type="multihead",
        )
        self.linear_in = torch.nn.Linear(in_channels, hidden_channels)
        self.linear_out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = self.linear_in(x)
        x = self.gps1(x, edge_index, batch)
        x = self.gps2(x, edge_index, batch)
        x = self.linear_out(x)
        return x
    
    def get_embeddings(self, x, edge_index, batch=None):
        x = self.linear_in(x)
        x = self.gps1(x, edge_index, batch)
        embeddings = self.gps2(x, edge_index, batch)
        return embeddings