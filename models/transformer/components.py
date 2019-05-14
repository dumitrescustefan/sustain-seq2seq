import torch.nn as nn
import torch
import math


class AttentionHead(nn.Module):
    def __init__(self, d_model, d_k, d_v, device):
        super(AttentionHead, self).__init__()
        self.dk = math.sqrt(d_k)

        self.query_layer = nn.Linear(d_model, d_k)
        self.key_layer = nn.Linear(d_model, d_k)
        self.value_layer = nn.Linear(d_model, d_v)
        self.to(device)

    def forward(self, input_query, input_key, input_value):
        query = self.query_layer(input_query)
        key = torch.transpose(self.key_layer(input_key), 1, 2)
        value = self.value_layer(input_value)

        score = torch.matmul(query, key)
        score = torch.nn.functional.softmax(score, dim=2)

        z = torch.matmul(score, value)

        return z


class Attention(nn.Module):
    def __init__(self, d_model, h, d_k, d_v, device):
        super(Attention, self).__init__()
        self.h = h

        self.heads = [AttentionHead(d_model, d_k, d_v, device) for _ in range(h)]

        self.linear = nn.Linear(h * d_v, d_model)
        self.to(device)
        
    def forward(self, input_query, input_key, input_value):
        for i in range(self.h):
            if i == 0:
                z = self.heads[i].forward(input_query, input_key, input_value)
            else:
                z = torch.cat((z, self.heads[i].forward(input_query, input_key, input_value)), dim=2)

        z = self.linear(z)

        return z

    def eval(self):
        super().eval()

        for i in range(self.h):
            self.heads[i].eval()

    def train(self, mode=True):
        super().train(mode)

        for i in range(self.h):
            self.heads[i].train(mode)

    def to(self, device):
        super().to(device)

        for i in range(self.h):
            self.heads[i].to(device)

