import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionNetwork(nn.Module):

    def __init__(self, input_size):
        super(AttentionNetwork, self).__init__()
        self.dense_layer = nn.Linear(input_size, 1)
        
    def forward(self, vertex_matrix, adjacency_matrix):
        vertex_normalized = torch.sigmoid(vertex_matrix)
        mu = torch.matmul(adjacency_matrix, vertex_normalized)
        mu_self = torch.diag(mu)
        mu_sum = torch.sum(mu, dim=1) - mu_self
        
        attention_logits = self.dense_layer(vertex_normalized)
        attention_logits += mu_self.view(-1, 1) + mu_sum.view(-1, 1)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        return attention_weights