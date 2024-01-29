import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from config import args
from util import AverageMeter


class R2D2:
    def __init__(self, init_scale=1e-4, init_bias=0, learn_lambda=False, init_lambda=1):
        self.adjust_layer = AdjustLayer(init_scale, init_bias).cuda()
        self.lambda_layer = LambdaLayer(learn_lambda, init_lambda).cuda()
        self.W = None
        self.loss_stat = AverageMeter()
        self.losses = []

    def fit(self, support, support_ys, mode='train'):
        n = support.shape[0]
        shuffled_ids = np.arange(n)
        np.random.shuffle(shuffled_ids)

        support_ys_ohe = torch.zeros((support_ys.shape[0], args.n_ways), dtype=torch.float32)
        support_ys_ohe[torch.arange(25), support_ys] = 1.
        support_ys_ohe = support_ys_ohe[shuffled_ids]

        X = torch.cat((support, torch.ones((support.shape[0], 1))), axis=1)
        X = X[shuffled_ids]

        if mode == 'train':
            self.lambda_layer.train()
            self.W = X.T @ torch.inverse(X @ X.T + self.lambda_layer(torch.eye(n))) @ support_ys_ohe
        else:
            self.lambda_layer.eval()
            with torch.no_grad():
                self.W = X.T @ torch.inverse(X @ X.T + self.lambda_layer(torch.eye(n))) @ support_ys_ohe

    def predict(self, query, mode='train'):
        query_features = torch.cat((query, torch.ones((query.shape[0], 1))), axis=1)
        if mode == 'train':
            self.adjust_layer.train()
            logits = self.adjust_layer(query_features @ self.W)
        else:
            self.adjust_layer.eval()
            with torch.no_grad():
                logits = self.adjust_layer(query_features @ self.W)
        return logits


class AdjustLayer(nn.Module):
    def __init__(self, init_scale=1e-4, init_bias=0, base=1):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_scale]))
        self.bias = nn.Parameter(torch.FloatTensor([init_bias]))
        self.base = base

    def forward(self, x):
        if self.base == 1:
            return x * self.scale + self.bias
        else:
            return x * (self.base ** self.scale) + self.base ** self.bias - 1


class LambdaLayer(nn.Module):
    def __init__(self, learn_lambda=False, init_lambda=1, base=1):
        super().__init__()
        self.l = torch.FloatTensor([init_lambda])
        self.base = base
        if learn_lambda:
            self.l = nn.Parameter(self.l)
        else:
            self.l = Variable(self.l)

    def forward(self, x):
        if self.base == 1:
            return x * self.l
        else:
            return x * (self.base ** self.l)