from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import LSTM
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model

from google.colab import files
uploaded = files.upload()
import io
data = pd.read_csv(io.StringIO(uploaded['data.csv'].decode('utf-8')))
# data = pd.read_csv('data.csv')

relevant_columns = np.array(data.columns.isin(['GEO', 'REF_DATE', 'DGUID', 'total_house_land'])) #+ np.array(data.dtypes != object)
input = data.loc[:, relevant_columns]
print(input.columns)

province_dguid = ['2016A000235', '2016A000259', '2016A000224', '2016A000248', '2016A000212', '2016A000246', '2016A000247']
# ON, BC, QB, AL, NS, MT

input1 = input[input['DGUID'].isin(province_dguid)]
province_data = pd.concat([input1, pd.get_dummies(input1.GEO, prefix='Province')], axis=1).sort_values(['GEO', 'REF_DATE'])
# province_data = input1.sort_values(['GEO', 'REF_DATE'])
print(province_data.columns)

feature_columns = province_data.columns[np.array(province_data.dtypes != object)]
target_column = 'total_house_land'
print(feature_columns)

def drop_data(df):
    tmp = df.dropna(subset=[target_column])
    return tmp

result = drop_data(province_data)
print(result.info())

def create_dataset(df, look_back=1):
  dataX, dataY, dataP, dataT, dataD = [], [], [], [], []
  for dguid in province_dguid:
    dataset = df[df['DGUID']== dguid]
    for i in range(len(dataset)-look_back):
      a = dataset[feature_columns][i:(i+look_back)]
      dataX.append(np.array(a))
      dataY.append(np.array(dataset[target_column].iloc[i + look_back]))
      dataT.append(np.array(dataset['REF_DATE'].iloc[i + look_back]))
      dataP.append(np.array(dataset['GEO'].iloc[i + look_back]))
      dataD.append(np.array(dataset['DGUID'].iloc[i + look_back]))
  return np.array(dataX), np.array(dataY), np.array(dataT), np.array(dataP), np.array(dataD)

# reshape into X=t and Y=t+1
look_back = 6
dataX, dataY, dataT, dataP, dataD = create_dataset(result, look_back)

print(dataX[2708], dataY[2708], dataT[2708], dataP[2708], dataD[2708])

print(result.shape)
print(np.shape(dataX), np.shape(dataY))
last_idx = np.shape(dataY)[0] - 1
print(dataX[last_idx,:,:], dataY[last_idx], dataT[last_idx], dataP[last_idx], dataD[last_idx])

def train_valid_split(X, Y, time, geo, dguid, year):
    trainX = X[time < year + '-01']
    trainY = Y[time < year + '-01']
    trainT = time[time < year + '-01']
    trainP = geo[time < year + '-01']
    trainD = dguid[time < year + '-01']
    validX = X[time >= year + '-01']
    validY = Y[time >= year + '-01']
    validT = time[time >= year + '-01']
    validP = geo[time >= year + '-01']
    validD = dguid[time >= year + '-01']
    return [trainX, trainY, trainT, trainP, trainD, validX, validY, validT, validP, validD]
 
trainX, trainY, trainT, trainP, trainD, validX, validY, validT, validP, validD = train_valid_split(dataX, dataY, dataT, dataP, dataD, '2018')

# create and fit the LSTM network

def basic_model_lstm(look_back):
    t_model = Sequential()
    t_model.add(LSTM(256, return_sequences=False, input_shape=(look_back, len(feature_columns))))
#     t_model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.15, return_sequences=True))
#     t_model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.15))
    t_model.add(Dense(1))
    t_model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=[metrics.mae])
    return(t_model)

model = basic_model_lstm(look_back)
model.summary()

epochs = 50
batch_size = 32

print('Epochs: ', epochs)
print('Batch size: ', batch_size)

history = model.fit(trainX, trainY,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=1, # Change it to 2, if wished to observe execution
    validation_data=(validX, validY))

train_score = model.evaluate(trainX, trainY, verbose=0)
valid_score = model.evaluate(validX, validY, verbose=0)

print('Train MAE: ', round(train_score[1], 4), ', Train Loss: ', round(train_score[0], 4)) 
print('Val MAE: ', round(valid_score[1], 4), ', Val Loss: ', round(valid_score[0], 4))

# valid_predict = model.predict(validX)
valid_predict = model.predict(validX)
print(valid_predict.shape)

# Produce a plot for the results.
for dguid in province_dguid:
  plt.plot(validT[validD == dguid], validY[validD == dguid])
  plt.plot(validT[validD == dguid], valid_predict[validD == dguid])
  plt.ylabel('Total Housing Index')
  plt.legend(['Actual Index','Predicted Index'])
  plt.title('LSTM Model - {}'.format(validP[validD == dguid][0]))
  plt.xlabel('Time')
  plt.show()

# Produce a plot for the results.
var = []
for i in validX:
  print(i)
  var.append(model.predict(i.reshape(1, i.shape[0], i.shape[1])))

valid_predict = np.array(var).reshape(np.shape(var)[0], np.shape(var)[1])
print(valid_predict.shape)
  
for dguid in province_dguid:
  plt.plot(validT[validD == dguid], validY[validD == dguid])
  plt.plot(validT[validD == dguid], valid_predict[validD == dguid])
  plt.ylabel('Total Housing Index')
  plt.legend(['Actual Index','Predicted Index'])
  plt.title('LSTM Model - {}'.format(validP[validD == dguid][0]))
  plt.xlabel('Time')
  plt.show()



