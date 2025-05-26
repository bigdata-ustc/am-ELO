# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser(description='KCAT')

parser.add_argument('--key', type=str, default='xxx', help="huggingface key")
params = parser.parse_args()