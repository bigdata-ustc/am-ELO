# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser(description='KCAT')

parser.add_argument('--device', type=str, default='cuda', help='cuda, cpu')
parser.add_argument('--seed', type=int, default=2024, help='the random seed,2023,2024,2025,2026,2027')
parser.add_argument('--data_path', type=str, default='data/chatbot.csv', help='data path')
parser.add_argument('--learning_rate', type=float, default=1e-1, help='learning rate')
parser.add_argument('--method', type=str, default='m-ELO', help="estimation method: ELO,m-ELO,am-ELO")
params = parser.parse_args()