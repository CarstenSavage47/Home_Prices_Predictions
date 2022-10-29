# Thank you to Christian Versloot for help with implementing a linear regression in PyTorch.
# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md

import pandas
import openpyxl
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

HomePrices = pandas.read_csv('/Users/carstenjuliansavage/PycharmProjects/RE_Regression/House Prices.csv')

## To be continued...
