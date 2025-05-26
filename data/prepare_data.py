# -*- coding: utf-8 -*-

import pandas as pd
from huggingface_hub import login
from setting import *

if __name__ == "__main__":
    login(token=params.key)
    df = pd.read_parquet("hf://datasets/lmsys/chatbot_arena_conversations/data/train-00000-of-00001-cced8514c7ed782a.parquet")
    df.to_csv("Chatbot.csv")