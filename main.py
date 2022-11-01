'''
Thank you to L1aoXingyu for help with implementing a linear regression in PyTorch.
https://github.com/L1aoXingyu/pytorch-beginner/blob/master/01-Linear%20Regression/Linear_Regression.py
'''
import pandas
import numpy as np
import openpyxl
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import SGD
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from tqdm import tqdm
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import plotly.express as px

HomePrices = pandas.read_csv('/Users/carstenjuliansavage/PycharmProjects/RE_Regression/House Prices.csv')

''' Group columns by datatypes for inspection '''
Column_VarType_Dict = HomePrices.columns.to_series().groupby(HomePrices.dtypes).groups
Column_VarType_Dict

''' I'm going to make the year vars here strings/objects '''
HomePrices = (HomePrices
              .astype({"YearBuilt":'str',"YearRemodAdd":'str',"GarageYrBlt":'str',"YrSold":'str'})
              )

HomePrices = pandas.get_dummies(HomePrices)

''' I needed to get rid of NA values here, otherwise I'd get nan values for loss. '''
HomePrices = HomePrices.dropna()

X = np.array(HomePrices.drop(['SalePrice'],axis=1), dtype=np.float32)
y = np.array(HomePrices['SalePrice'], dtype=np.float32)

y = y.reshape(-1, 1)

# Split dataframe into training and testing data. Remember to set a seed.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)
#y = min_max_scaler.fit_transform(y)
''' Don't scale the depedent var. '''

# Converting X and y to tensors for part 1
X_TRTensor = torch.from_numpy(X_train)
X_TETensor = torch.from_numpy(X_test)
y_TRTensor = torch.from_numpy(y_train)
y_TETensor = torch.from_numpy(y_test)

N_Features = X_TRTensor.size(dim=1)

# Linear Regression Model
class linearRegression(nn.Module):
    def __init__(self):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(N_Features, 1)  # input -- 561 features and output -- Matrix with 1 column
    ''' A mxn matrix with 1195 rows and 561 columns (1195x561)
        ...must be multiplied with a mxn matrix with 561 rows and 1 column (561x1)
        That results in a mxn matrix with dimensions 1195x1.
    '''

    def forward(self, x):
        out = self.linear(x)
        return out


LinearRegression = linearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(LinearRegression.parameters(), lr=.01) # Optimal learning rate
'''
Remember - Errors mainly refer to difference between actual observed sample values and predicted values,
Residuals refer exclusively to the differences between dependent variables and estimations from linear regression.
'''

num_epochs = 100000
for epoch in range(num_epochs):
    inputs = X_TRTensor
    target = y_TRTensor

    # forward
    out = LinearRegression(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 500 == 0:
        print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}')

LinearRegression.eval()
with torch.no_grad():
    predict = LinearRegression(X_TETensor)
    predict = predict.data.numpy()

torch.save(LinearRegression.state_dict(), './linear.pth')

'''
# Compare observed - predicted
Comparison = (Comparison
              .filter(['SalePrice','Predicted_Price'])
              .assign(Difference = lambda a: round(a.SalePrice - a.Predicted_Price,2))
              .assign(ABSDifference=lambda a: round(abs(a.SalePrice - a.Predicted_Price), 2))
              .assign(Predicted_Price=lambda a: round(a.Predicted_Price))
              #.astype({"Predicted_Price":'int',"Difference":'int'})
              )

print('Mean Difference:',Comparison.agg({"Difference":'mean'}),'Mean ABS_Difference:',
      Comparison.agg({"ABSDifference":'mean'}))


''' Simulating addition of new data '''
Shuffled_Data = HomePrices.sample(frac=1/2)
X_Shuffled = np.array(Shuffled_Data.drop(['SalePrice'],axis=1), dtype=np.float32)
X_Shuffled = min_max_scaler.fit_transform(X_Shuffled)
# Converting X to tensors
X_Shuffled_Tensor = torch.from_numpy(X_Shuffled)

''' Testing new data '''
## Disable grad
with torch.no_grad():
    print('True SalePrice is:',(Shuffled_Data['SalePrice']).iloc[1])
    prediction = LinearRegression(X_Shuffled_Tensor[1])
    print('Predicted SalePrice is:',prediction)
    print('ABS Difference between observed and predicted:',
          abs((Shuffled_Data['SalePrice']).iloc[1] - prediction.numpy()))