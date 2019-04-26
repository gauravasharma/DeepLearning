import numpy as np
import pandas as pd

print("Process Started")
# url to download the data set -- https://archive.ics.uci.edu/ml/datasets/automobile
# Data set is in CSV format.
print("Load DataSet ")
automobile_data= pd.read_csv('DataSets/AutoMobile_data.csv',
    sep=r'\s*,\s*',engine='python')

print("Processing data and removing missing Values ")
#print(automobile_data.head())
#There are missing values denoted by question mark. Replace them by NAN.
automobile_data=automobile_data.replace('?',np.nan)

#Now drop records having NAN.
automobile_data=automobile_data.dropna() 
#print(automobile_data.head())

print("Selecting features")
col=['make','fuel-type','body-style','horsepower']
automobile_features=automobile_data[col]
automobile_target=automobile_data[['price']]
#print(automobile_features.head())

#print(automobile_features['horsepower'].describe())

pd.options.mode.chained_assignment=None

#pd.to_numeric converts the value to numeric
automobile_features['horsepower']=\
                   pd.to_numeric(automobile_features['horsepower']  )

#print(automobile_features['horsepower'].describe())

automobile_target= automobile_target.astype(float)
#print(automobile_target['price'].describe())

print("Perform one hot encoding")

#perfrom one hot encoding
automobile_features= pd.get_dummies(automobile_features, columns=['make','fuel-type','body-style'])
##print(automobile_features.head())
# print(automobile_features.columns)

#Before feeding it to the neural network we will do some preprocessing usaing sklearn.

from sklearn  import preprocessing

#Standardized the numeric values as ML Works better on the numbers.
# Subtract the Mean and divide by the standard deviation.
# Helps in solving Grdient Vanishing and exploding problem
# other ways 1- Proper intialization 2- Non Sauration Activation Function 3- Gradient Clipping
automobile_features['horsepower']=\
                   preprocessing.scale(automobile_features['horsepower']  )
# print(automobile_features['horsepower'].head())

print("split the dataset")
# split the dataset
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(automobile_features,
                                                 automobile_target,
                                                 test_size=0.2,
                                                 random_state=0)

print("Creating Tensors")
# PyTorch code starts here
import torch
dtype=torch.float

x_train_tensor = torch.tensor(x_train.values,dtype=dtype)
x_test_tensor= torch.tensor(x_test.values,dtype=dtype)

y_train_tensor = torch.tensor(y_train.values,dtype=dtype)
y_test_tensor= torch.tensor(y_test.values,dtype=dtype)


#print(x_train_tensor.shape)

print("Creeating fully connected neural network")

inp=26 # no of input features
out=1 # one output feature we predict that is price.
hid=100 # number of neurons in each layer
 # loss function Mean Square Error loss for Regression problem. Automatically calculated
loss_fn=torch.nn.MSELoss()
learning_rate=0.0001 # how bigger step we want to take in direction of reducing gradient.

# Creating Neural Network with 2 Layers and 100 neurons in each layer.
# Using Sigmoid fnction as an Activation Function
model=torch.nn.Sequential(torch.nn.Linear(inp,hid),
                          torch.nn.Sigmoid(),
                          torch.nn.Linear(hid,out)) 

print("Training the model now. Output of this step will be trained model.")
# # Total no of Epoch is 10000
for iter in range(10000):
  y_pred= model(x_train_tensor)
  loss=loss_fn(y_pred,y_train_tensor)

  if iter % 1000==0:
      print(iter,loss.item())
  model.zero_grad() # Zeros out the Calculated Grad from previous step if any.
  loss.backward() # This calculates the Gradient using the AutoGrad library(balcward pass)

  with torch.no_grad(): # turns off the trakin gfeature used by autograd.
       for param in model.parameters():
           param-=learning_rate*param.grad # for now updating the model parameter manually. Can use optimizer do the same.


# model is Trained now, Checking the one sample data from test data.
sample= x_test.iloc[23]
#print(sample)

sample_tensor= torch.tensor(sample.values, dtype= dtype)
#print(sample_tensor)

y_pred= model(sample_tensor)

print("Predicted price of automobile is:", int(y_pred.item()))
print("Actual price of automobile is:", int(y_test.iloc[23].item()))

y_pred_tensor= model(x_test_tensor)

y_pred=y_pred_tensor.detach().numpy()

##graph showing Acctual vs Prediction

import matplotlib.pyplot as plt

# plt.scatter(y_pred,y_test.values)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Predicted vs Actual Price.")
# plt.show()

## save the trained model to file system.
torch.save(model,'my_model')
saved_model= torch.load('my_model')
y_pred_tensor=saved_model(x_test_tensor)
y_pred=y_pred_tensor.detach().numpy()

plt.figure(figsize=(15,6))
plt.plot(y_pred, label="Predicted Price")
plt.plot(y_test.values,label="Actual Price")
plt.legend()
plt.title("Predicted vs Actual Price")
plt.show()

print('SuccessFully Completed')
