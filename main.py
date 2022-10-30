'''
Thank you to Christian Versloot for help with implementing a linear regression in PyTorch.
https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md
'''
import pandas
import numpy as np
import openpyxl
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torch  # torch provides basic functions, from setting a random seed (for reproducability) to creating tensors.
import torch.nn.functional as F  # nn.functional give us access to the activation and loss functions.
from torch.optim import SGD  # optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt  ## matplotlib allows us to draw graphs.
import seaborn as sns  ## seaborn makes it easier to draw nice-looking graphs.
import os
from tqdm import tqdm
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import plotly.express as px

HomePrices = pandas.read_csv('/Users/carstenjuliansavage/PycharmProjects/RE_Regression/House Prices.csv')

# Group columns by datatypes for inspection
Column_VarType_Dict = HomePrices.columns.to_series().groupby(HomePrices.dtypes).groups
Column_VarType_Dict

HomePrices = pandas.get_dummies(HomePrices)

X = HomePrices.drop(['SalePrice'],axis=1)
y = HomePrices['SalePrice']


