import numpy as np
import pandas as pd

print("Process Started")

print("Load Test DataSet ")
automobile_data= pd.read_csv('DataSets/AutoMobile_data_1.csv',
    sep=r'\s*,\s*',engine='python')

automobile_data=automobile_data.replace('?',np.nan)
automobile_data=automobile_data.dropna() 

print("Selecting features")
col=['make','fuel-type','body-style','horsepower']
automobile_features=automobile_data[col]

pd.options.mode.chained_assignment=None

#pd.to_numeric converts the value to numeric
automobile_features['horsepower']=\
                   pd.to_numeric(automobile_features['horsepower']  )

print("Perform one hot encoding")

#perfrom one hot encoding
automobile_features= pd.get_dummies(automobile_features, columns=['make','fuel-type','body-style'])
#Before feeding it to the neural network we will do some preprocessing usaing sklearn.

from sklearn  import preprocessing
automobile_features['horsepower']=\
                   preprocessing.scale(automobile_features['horsepower']  )

print("Creating Tensors")
# PyTorch code starts here
import torch
dtype=torch.float

# model is Trained now, Checking the one sample data from test data.

row=automobile_data.iloc[2]
print(row)
print(row[['price']])

sample= automobile_features.iloc[2]
print(sample)
sample_tensor= torch.tensor(sample.values, dtype= dtype)

saved_model= torch.load('my_model')
y_pred= saved_model(sample_tensor)

print("Predicted price of automobile is:", int(y_pred.item()))
#print("Actual price of automobile is:", int(y_test.iloc[23].item()))

print('SuccessFully Completed')
