# -*- coding: utf-8 -*-

import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pandas as pd
from utils import *
from setting import *

class am_ELO(nn.Module):
    def __init__(self,num_model,num_judge):
        super(am_ELO, self).__init__() 
        self.R = nn.Embedding(num_model,1)
        self.Theta = nn.Embedding(num_judge,1)
        torch.nn.init.xavier_uniform_(self.R.weight.data)
        torch.nn.init.xavier_uniform_(self.Theta.weight.data)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.device = torch.device("cuda") if params.device=='cuda' else torch.device("cpu")
        #self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        i,j,k = x[:,0],x[:,1],x[:,2]
        R_i = self.R(i)
        R_j = self.R(j)
        #theta = self.Theta(k)
        #theta = torch.exp(theta)/torch.exp(self.Theta.weight).sum()
        #theta = self.Theta(k)
        theta = self.Theta(k)/self.Theta.weight.sum()
        p = theta*(R_i-R_j)
        #p = torch.log(torch.tensor(10))/400*(R_i-R_j)
        return p
    
    def get_RA(self,train_data):
        x_train = train_data[['model_a','model_b','judge']]
        y_train = train_data['label']
        train_set=TensorDataset(torch.from_numpy(x_train.values),torch.from_numpy(y_train.values).to(torch.float))
        train_loader = DataLoader(train_set,batch_size=len(x_train), shuffle=True)
        
        train_loss = np.inf
        for num in range(2000):
            for data in train_loader:
                self.optimizer.zero_grad()#每次记得将梯度清零，否则梯度会累计。
                x, targets = data
                x = x.to(self.device)
                targets = targets.to(self.device)
                y_pred = self(x)
                loss = self.loss_fn(y_pred.reshape(-1), targets)
                loss.backward()
                self.optimizer.step()
        Theta = self.Theta.weight.detach().cpu().numpy().flatten()
        return self.R.weight.flatten().detach().cpu().numpy(), Theta/Theta.sum()

class m_ELO(nn.Module):
    def __init__(self,num_model,num_judge):
        super(m_ELO, self).__init__() 
        self.R = nn.Embedding(num_model,1)
        self.Theta = nn.Embedding(num_judge,1)
        torch.nn.init.xavier_uniform_(self.R.weight.data)
        torch.nn.init.xavier_uniform_(self.Theta.weight.data)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params.learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.device = torch.device("cuda") if params.device=='cuda' else torch.device("cpu")
        #self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        i,j = x[:,0],x[:,1]
        R_i = self.R(i)
        R_j = self.R(j)
        p = torch.log(torch.tensor(10))/400*(R_i-R_j)
        return p
    
    def get_RA(self,train_data):
        x_train = train_data[['model_a','model_b','judge']]
        y_train = train_data['label']
        train_set=TensorDataset(torch.from_numpy(x_train.values),torch.from_numpy(y_train.values).to(torch.float))
        train_loader = DataLoader(train_set,batch_size=len(x_train), shuffle=True)
        train_loss = np.inf
        for num in range(2000):
            for data in train_loader:
                self.optimizer.zero_grad()#每次记得将梯度清零，否则梯度会累计。
                x, targets = data

                x = x.to(self.device)
                targets = targets.to(self.device)
                y_pred = self(x)
                loss = self.loss_fn(y_pred.reshape(-1), targets)
                loss.backward()
                self.optimizer.step()
        return self.R.weight.flatten().detach().cpu().numpy(), ''

class Traditional_ELO:
    def compute_elo(self,battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
        self.rating = defaultdict(lambda: INIT_RATING)

        for rd, model_a, model_b, winner in battles[['model_a', 'model_b', 'winner']].itertuples():
            ra = self.rating[model_a]
            rb = self.rating[model_b]
            ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
            eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
            if winner == "model_a":
                sa = 1
            elif winner == "model_b":
                sa = 0
            elif winner == "tie" or winner == "tie (bothbad)":
                sa = 0.5
            else:
                raise Exception(f"unexpected vote {winner}")
            self.rating[model_a] += K * (sa - ea)
            self.rating[model_b] += K * (1 - sa - eb)

        return self.rating
    
    def preety_print_elo_ratings(self):
        df = pd.DataFrame([
            [n, self.rating[n]] for n in self.rating.keys()
        ], columns=["Model", "Elo rating"]).sort_values("Elo rating", ascending=False).reset_index(drop=True)
        df["Elo rating"] = (df["Elo rating"] + 0.5).astype(int)
        df.index = df.index + 1
        return df
    def get_bootstrap_result(self, battles, num_round):
        rows = []
        for i in range(num_round):
            rows.append(self.compute_elo(battles.sample(frac=1.0, replace=True)))
        df = pd.DataFrame(rows)
        bootstrap_elo_lu = df[df.median().sort_values(ascending=False).index]
        bootstrap_lu_median = bootstrap_elo_lu.median().reset_index().set_axis(["model", "ELO Score"], axis=1)
        return bootstrap_lu_median