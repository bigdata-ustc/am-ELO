# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from utils import *
from model import *
from setting import *
class Dataset(object):

    def __init__(self, data_path):
        self.model2index = {}
        self.judge2index = {}
        self.index2model = []
        self.index2judge = []
        
        df = pd.read_csv(data_path)
        self.df = self.filterd(df)
        self.data = self.df2dict(self.df)
        self.model_num = len(self.df['model_a'].unique())
        self.anno_num = len(self.df['judge'].unique())
        params.model_num = self.model_num
        params.anno_num = self.anno_num
    
    def filterd(self,df):
        values=df['judge'].value_counts()
        keys = df['judge'].value_counts().keys()
        
        judge =[]
        for k in keys:
            if values[k]>50:
                judge.append(k)
        df['is_true'] = df.apply(lambda x:(x['judge'] in judge),axis=1)
        df = df[df['is_true']]
        df = df.drop('is_true', axis=1)
        return df
    
    def df2dict(self,df):
        data = {'model_a':[],'model_b':[],'judge':[],'label':[]}
        m = 0
        n = 0
        for index,item in df.iterrows():
            if item['model_a'] not in self.model2index.keys():
                self.model2index[item['model_a']] = m
                self.index2model.append(item['model_a'])
                m += 1
            if item['model_b'] not in self.model2index.keys():
                self.model2index[item['model_b']] = m
                self.index2model.append(item['model_b'])
                m += 1
            if item['judge'] not in self.judge2index.keys():
                self.judge2index[item['judge']] = n
                self.index2judge.append(item['judge'])
                n += 1
            data['model_a'].append(self.model2index[item['model_a']])
            data['model_b'].append(self.model2index[item['model_b']])
            data['judge'].append(self.judge2index[item['judge']])
            winner = item['winner']
            if winner == "model_a":
                sa = 1
            elif winner == "model_b":
                sa = 0
            elif winner == "tie" or winner == "tie (bothbad)":
                sa = 0.5
            data['label'].append(sa)
        return data
    
    def get_ELO(self,method):
        if method == 'am-ELO':
            net = am_ELO(self.model_num,self.anno_num).to(self.device)
        if method == 'm-ELO':
            net = m_ELO(self.model_num,self.anno_num).to(self.device)
        R_star,A_star = net.get_RA(pd.DataFrame(self.data))
        seed_torch(2025)
        A_list = np.array(self.data['judge'])
        np.random.shuffle(A_list)
        return R_star, A_star,A_list
    
    def get_dataframe(self, method):
        if method == 'ELO':
            net = Traditional_ELO()
            model2elo = net.get_bootstrap_result(self.df, 1000)
            return model2elo, ''
        else:
            self.device = torch.device("cuda") if params.device=='cuda' else torch.device("cpu")
            self.R_star, self.A_star, self.A_list = self.get_ELO(method)
            model2elo = {'model':[],'ELO_Score':[]}
            judge2elo = {'judge':[],'ELO_Score':[]}
            R_rank = np.argsort(self.R_star)[::-1]
            for i in R_rank:
                model2elo['model'].append(self.index2model[i])
                model2elo['ELO_Score'].append(self.R_star[i])
            if method == 'm-ELO':
                return pd.DataFrame(model2elo), ''
            A_rank = np.argsort(self.A_star)[::-1]
            for i in A_rank:
                judge2elo['judge'].append(self.index2judge[i])
                judge2elo['ELO_Score'].append(self.A_star[i])
        return pd.DataFrame(model2elo),pd.DataFrame(judge2elo)