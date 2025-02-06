#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='', name='Cora')
data = dataset[0]


# In[2]:


print(f'Dataset: {dataset}')
print("----------------------------------------")
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of features: {dataset.num_features}')
print('Number of classes: {dataset.num_classes}')

print(f'\nGraph:')
print('----------------------------------------')
print(f'Edges are directed: {data.is_directed()}')
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
print(f'Graph has loops: {data.has_self_loops()}')


# In[3]:


from torch_geometric.utils import to_dense_adj

adjacency = to_dense_adj(data.edge_index)[0]
adjacency += torch.eye(len(adjacency))
adjacency


# In[4]:


from torch.nn import Linear


class GNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = Linear(dim_in, dim_out, bias=False)

    def forward(self, x, adjacency):
        x = self.linear(x)

        x = torch.sparse.mm(adjacency, x)
        return x


# In[5]:


import torch
import torch.nn.functional as F
from torchmetrics.functional.classification.accuracy import accuracy


class GNN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gnn1 = GNNLayer(dim_in, dim_h)
        self.gnn2 = GNNLayer(dim_h, dim_out)

    def forward(self, x, adjacency):
        h = self.gnn1(x, adjacency)
        h = torch.relu(h)
        h = self.gnn2(h, adjacency)
        return F.log_softmax(h, dim=1)

    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=0.01,
                                     weight_decay=5e-4)

        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, adjacency)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            if (epoch % 20 == 0):
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim=1),
                                   data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc:'
                      f' {acc * 100:>5.2f}% | Val Loss: {val_loss:.2f} | '
                      f'Val Acc: {val_acc * 100:.2f}%')

    @torch.no_grad
    def test(self, data):
        self.eval()
        out = self(data.x, adjacency)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc





# In[6]:


def accuracy(preds, targets):
    """
    Args:
        preds (torch.Tensor): 예측값 (예: argmax 결과), shape: (N,)
        targets (torch.Tensor): 실제 레이블, shape: (N,)
    
    Returns:
        float: 정확도 (0~1 사이의 값)
    """
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


# In[7]:


def accuracy(preds, targets):
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


# In[8]:


gnn = GNN(dataset.num_features, 16, dataset.num_classes)
print(gnn)

# Train
gnn.fit(data, epochs=100)

# Test
acc = gnn.test(data)
print(f'\nGNN test accuracy: {acc * 100:.2f}%')


# In[9]:


from torch_geometric.datasets import FacebookPagePage

dataset = FacebookPagePage(root=".")
data = dataset[0]
data.train_mask = range(18000)
data.val_mask = range(18001, 20000)
data.test_mask = range(20001, 22470)

# Adjacency matrix
adjacency = to_dense_adj(data.edge_index)[0]
adjacency += torch.eye(len(adjacency))
adjacency


# GNN
gnn = GNN(dataset.num_features, 16, dataset.num_classes)
print(gnn)
gnn.fit(data, epochs=100)
acc = gnn.test(data)
print(f'\nGNN test accuracy: {acc*100:.2f}%')


# In[ ]:




