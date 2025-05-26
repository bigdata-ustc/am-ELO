# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
from dataset import *
from utils import *
from setting import *


if __name__ == '__main__':
    device = torch.device("cuda") if params.device=='cuda' else torch.device("cpu")
    params.C = np.log(10)/400
    dataset = Dataset(params.data_path) 
    model2elo,judge2elo = dataset.get_dataframe(params.method)
    print(model2elo)
    if params.method=='am-ELO':
        print(judge2elo)
    