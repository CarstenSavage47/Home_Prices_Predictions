'''
Thank you to L1aoXingyu for help with implementing a linear regression in PyTorch.
https://github.com/L1aoXingyu/pytorch-beginner/blob/master/01-Linear%20Regression/Linear_Regression.py
'''
import pandas
import numpy as np
import openpyxl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch  # torch provides basic functions, from setting a random seed (for reproducability) to creating tensors.
import torch.nn.functional as F  # nn.functional give us access to the activation and loss functions.
from torch.optim import SGD  # optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
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

# I needed to get rid of NA values here, otherwise I'd get nan values for loss.
HomePrices = HomePrices.dropna()

X = np.array(HomePrices.drop(['SalePrice'],axis=1), dtype=np.float32)
y = np.array(HomePrices['SalePrice'], dtype=np.float32)

y = y.reshape(-1, 1)

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
y = min_max_scaler.fit_transform(y)

X = torch.from_numpy(X)
y = torch.from_numpy(y)

# Linear Regression Model
class linearRegression(nn.Module):
    def __init__(self):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(289, 1)  # input -- 289 features and output -- 1 feature

    def forward(self, x):
        out = self.linear(x)
        return out


LinearRegression = linearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(LinearRegression.parameters(), lr=.01)

num_epochs = 1000
for epoch in range(num_epochs):
    inputs = X
    target = y

    # forward
    out = LinearRegression(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}')

LinearRegression.eval()
with torch.no_grad():
    predict = LinearRegression(X)
predict = predict.data.numpy()

fig = plt.figure(figsize=(10, 5))
plt.plot(X.numpy(), y.numpy(), 'ro', label='Original data')
plt.plot(X.numpy(), predict, label='Fitting Line')

plt.legend()
plt.show()

torch.save(LinearRegression.state_dict(), './linear.pth')