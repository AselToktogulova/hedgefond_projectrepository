import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from help_methods import create_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

look_back = 100

### model
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(look_back,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
###

df=pd.read_csv('./data/selected_stocks/WDI.DE_shorten.csv')
df1 = df.reset_index()['Close']
df1 = df1.dropna()

dataset_length = df1.shape[0]
look_back=100

scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

training_size=int(len(df1)*0.8)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

# reshaping (X=t,t+1,t+2,t+3, target: Y=t+4)
X_train, y_train = create_dataset(train_data, look_back)
X_test, ytest = create_dataset(test_data, look_back)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=50,batch_size=64,verbose=1)

### prediction
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
## transofrm back to original format
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### loss error RMSE
error_train = math.sqrt(mean_squared_error(y_train,train_predict))
### loss error RMSE
math.sqrt(mean_squared_error(ytest,test_predict))

trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1), color='red', label='real data')
plt.plot(trainPredictPlot, color="green", label='training data')
plt.plot(testPredictPlot, color="blue", label='evaluation data')

plt.xlabel('Date', fontsize=15)
plt.legend(prop={'size': 15})
plt.xticks(color='w')
plt.tight_layout()
plt.show()

x_input=test_data[len(test_data)-look_back:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

day_to_predict = 30
lst_output = []
n_steps = look_back
i = 0
while (i < day_to_predict):

    if (len(temp_input) > look_back):
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i = i + 1

temp = look_back
day_new=np.arange(1,temp)
day_pred=np.arange(temp-1,temp + day_to_predict)

plt.plot(day_new,scaler.inverse_transform(df1[dataset_length-(temp-1):]), color='black', label="passed through the model")

df=pd.read_csv('./data/selected_stocks/WDI.DE_shorten.csv')
df1 = df.reset_index()['Close']
df1 = df1.dropna()
last_day = df1.values.tolist()[len(df1.values.tolist())-1]

plt.plot(day_pred, [last_day] + scaler.inverse_transform(lst_output).T.tolist()[0], color='orange', label='predicted')

df=pd.read_csv('./data/selected_stocks/WDI-DE_tail.csv')
df1 = df.reset_index()['Close']
df1 = df1.dropna()

day_new=np.arange(temp-1,temp-1 + len(df1.values)+2)

plt.plot(day_new, [last_day] + df['Close'].values.tolist(), color='red', label='real data')

plt.xticks(color='w')
plt.legend(prop={'size': 15})
plt.xlabel('Date', fontsize=15)
plt.tight_layout()
plt.show()



