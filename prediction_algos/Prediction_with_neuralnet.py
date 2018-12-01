from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model

#from google.colab import files
#uploaded = files.upload()
#import io
#data = pd.read_csv(io.StringIO(uploaded['data.csv'].decode('utf-8')))
data = pd.read_csv('data.csv')
data['REF_DATE_INT'] = data.REF_DATE.map(lambda x: int(x.replace('-', '')))

relevant_columns = np.array(data.columns.isin(['GEO', 'REF_DATE', 'DGUID'])) + np.array(data.dtypes != object)
input = data.loc[:, relevant_columns]
print(input.columns)

province_dguid = '2016A000235', '2016A000259', '2016A000224', '2016A000248', '2016A000212', '2016A000246', '2016A000247'
# ON, BC, QB, AL, NS, MT

input1 = input[input['DGUID'].isin(province_dguid)]
province_data = pd.concat([input1, pd.get_dummies(input1.GEO, prefix='Province')], axis=1).sort_values(['REF_DATE_INT', 'GEO'])
print(province_data.columns)

feature_columns = province_data.columns[np.array(province_data.dtypes != object)][3:]
target_column = 'total_house_land'
print(feature_columns)

def drop_data(df):
    tmp = df.dropna(subset=[target_column])
    return tmp


def fill_na(data, null_columns, feature_columns):
    linreg = LinearRegression()
    for dguid in province_dguid:
      df = data[data['DGUID']== dguid]
      for col_name in null_columns:
          df_with_null = df[feature_columns + [col_name]]
          df_without_null = df_with_null.dropna()
          if(df_without_null[col_name].count() == 0):
            print("******** {} ********".format(col_name))
            df[col_name] = 0
            break
          x = df_without_null[feature_columns]
          y = df_without_null[col_name]
          linreg.fit(x, y)
          df_with_null['predicted'] = linreg.predict(df_with_null[feature_columns])
          df[col_name].fillna(df_with_null.predicted, inplace=True)
      data.update(df)
    return data

null_columns = ['income', 'food_expenditures', 'income_taxes', 'mortageinsurance', 'mortagePaid', 'accomodation', 'rent', 'shelter', 'total_expenditure', 'taxes_landregfees', 'international_tourism', 'domestic_tourism', 'employment', 'fulltime', 'labourforce', 'parttime', 'population', 'unemployment', 'employment_rate', 'participationrate', 'unemployment_rate', 'crime_incidents', 'cpi_index', 'diesel_fillingstations', 'diesel_selfservstations', 'premium_fillingstations', 'premium_selfservstations', 'regular_fillingstations', 'regular_selfservstations', 'immigrants', '1y_fixed_posted', '2y_bond', '3y_bond', '3y_fixed_posted', '5y_bond', '5y_fixed_posted', '7y_bond', '10y_bond', 'bank', 'overnight', 'overnight_target', 'prime', 'Mean_Max_Temp', 'Mean_Min_Temp', 'Mean_Temp', 'Total_Rain', 'Total_Snow']
target_columns = ['REF_DATE_INT'] # + [col for col in province_data if col.startswith('Province')]
result = fill_na(drop_data(province_data), null_columns, target_columns)

def train_valid_split(df, year):
    train = result[result.REF_DATE.map(lambda x: x < year + '-01')]
    valid = result[result.REF_DATE.map(lambda x: x >= year + '-01')]
    return [train, valid]

def min_max_scaler(df):
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

train, valid = train_valid_split(min_max_scaler(result), '2018')
# train, valid = train_valid_split(result, '2018')

print('Training shape:', train.shape)
print('Validation samples: ', valid.shape[0])

def basic_model_3(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(80, activation="tanh", kernel_initializer='normal', input_shape=(x_size,)))
    t_model.add(Dropout(0.2))
    t_model.add(Dense(120, activation="relu", kernel_initializer='normal',
        kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(20, activation="relu", kernel_initializer='normal',
        kernel_regularizer=regularizers.l1_l2(0.01), bias_regularizer=regularizers.l1_l2(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(10, activation="relu", kernel_initializer='normal'))
    t_model.add(Dropout(0.0))
    t_model.add(Dense(y_size))
    t_model.compile(
        loss='mean_squared_error',
        optimizer='nadam',
        metrics=[metrics.mae])
    return(t_model)

model = basic_model_3(train[feature_columns].shape[1], pd.DataFrame(train[target_column]).shape[1])
model.summary()

epochs = 500
batch_size = 128

print('Epochs: ', epochs)
print('Batch size: ', batch_size)

history = model.fit(train[feature_columns], pd.DataFrame(train[target_column]),
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=1, # Change it to 2, if wished to observe execution
    validation_data=(valid[feature_columns], pd.DataFrame(valid[target_column])))

train_score = model.evaluate(train[feature_columns], pd.DataFrame(train[target_column]), verbose=0)
valid_score = model.evaluate(valid[feature_columns], pd.DataFrame(valid[target_column]), verbose=0)

print('Train MAE: ', round(train_score[1], 4), ', Train Loss: ', round(train_score[0], 4))
print('Val MAE: ', round(valid_score[1], 4), ', Val Loss: ', round(valid_score[0], 4))

valid['predict'] = model.predict(valid[feature_columns])

# Produce a plot for the results.
for dguid in province_dguid:
  data = valid[valid['DGUID'] == dguid]
  plt.plot(np.array(data['REF_DATE']), data[target_column])
  plt.plot(np.array(data['REF_DATE']), data['predict'])
  plt.ylabel('Total Housing Index')
  plt.legend(['Actual Index','Predicted Index'])
  plt.title('Neural Network Model 3 - {}'.format(data.GEO.iloc[0]))
  plt.xlabel('Time')
  plt.show()