# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import keras_tuner
from keras import layers

# Fetching the data
data = pd.read_csv('daily_revenue.csv', sep='\t')
data.head()
'''
   Unnamed: 0      date  revenue discount_rate coupon_rate
0           1  01/01/18  6270839        34.27%       1.09%
1           2  02/01/18  8922076        30.87%       1.08%
2           3  03/01/18  8446101        28.11%       1.01%
3           4  04/01/18  7785798        27.32%       0.96%
4           5  05/01/18  6375303        25.70%       0.90%
'''

# Dropping the  "Unnamed: 0" column
data.drop('Unnamed: 0', axis=1, inplace=True)

# Converting the "date" column to a datetime column
dt = pd.to_datetime(data['date'], infer_datetime_format=True)
'''
Out[11]: 
0      2018-01-01
1      2018-02-01
2      2018-03-01
3      2018-04-01
4      2018-05-01
          ...    
1790   2022-11-26
1791   2022-11-27
1792   2022-11-28
1793   2022-11-29
1794   2022-11-30
Name: date, Length: 1795, dtype: datetime64[ns]
'''

# Joining the 2 dataframes
data = pd.concat([dt, data[['revenue', 'discount_rate', 'coupon_rate']]], axis=1)
data.set_index(['date'], inplace=True)
'''
            revenue discount_rate coupon_rate
date                                         
2018-01-01  6270839        34.27%       1.09%
2018-02-01  8922076        30.87%       1.08%
2018-03-01  8446101        28.11%       1.01%
2018-04-01  7785798        27.32%       0.96%
2018-05-01  6375303        25.70%       0.90%
'''

# Replacing the "%" from "discount_rate" and "revenue" columns
# Converting the values to integer and float values
data['discount_rate'] = data['discount_rate'].str.replace('%','')
data['coupon_rate'] = data['coupon_rate'].str.replace('%','')
convert_dict = {'revenue': int,
                'discount_rate': float,
                'coupon_rate' : float
                }
data = data.astype(convert_dict)
data['discount_rate'] = data['discount_rate'] / 100
data['coupon_rate'] = data['coupon_rate'] / 100
'''
            revenue  discount_rate  coupon_rate
date                                           
2018-01-01  6270839         0.3427       0.0109
2018-02-01  8922076         0.3087       0.0108
2018-03-01  8446101         0.2811       0.0101
2018-04-01  7785798         0.2732       0.0096
2018-05-01  6375303         0.2570       0.0090
'''

# Normalising the values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler = scaler.fit(data[['revenue', 'discount_rate', 'coupon_rate']])
data = scaler.transform(data[['revenue', 'discount_rate', 'coupon_rate']])

data = pd.DataFrame(data)
'''
          0         1         2
0  0.033234  1.000000  0.082204
1  0.069186  0.870919  0.081301
2  0.062732  0.766135  0.074977
3  0.053778  0.736143  0.070461
4  0.034650  0.674639  0.065041
'''

# Splitting the dataframe into train/test
df_test = data.tail(359)
df_train = data.head(1436)

# Setting the metric for calculation
from sklearn.metrics import mean_absolute_error

# Creating the dataset in the proper format
n_future = 30  # Number of days we want to look into the future based on the past days.
trainX = []
trainY = []
n_past = 30  # Number of past days we want to use to predict the future.
for i in range(n_past, len(df_train) - n_future + 1):
    dt = df_train.values
    dt = dt.astype('float32')
    trainX.append(dt[i - n_past:i, 0:dt.shape[1] - 1])
    trainY.append(dt[i - n_past:i, -1])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
'''
trainX shape == (1377, 30, 2).
trainY shape == (1377, 30).
'''

testX = []
testY = []

for i in range(n_past, len(df_test) - n_future + 1):
    dt = df_test.values
    dt = dt.astype('float32')
    testX.append(dt[i - n_past:i, 0:dt.shape[1] - 1])
    testY.append(dt[i - n_past:i, -1])

testX, testY = np.array(testX), np.array(testY)

print('testX shape == {}.'.format(testX.shape))
print('testY shape == {}.'.format(testY.shape))
'''
testX shape == (300, 30, 2).
testY shape == (300, 30).
'''

# Building the model
model = tf.keras.models.Sequential([tf.keras.layers.LSTM(units=16, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True),
                                   tf.keras.layers.Dense(1, activation='sigmoid')])
model.summary()
model.compile(optimizer='Adam', loss=tf.keras.losses.MeanAbsoluteError())
epochs_hist = model.fit(trainX, trainY, epochs = 100, batch_size = 50, validation_data=(testX, testY))

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

testPredict = model.predict(testX)
testPredict = np.reshape(testPredict, testY.shape)
print("Mean Absolute Error achieved : ", mean_absolute_error(testY, testPredict))
'''
Mean Absolute Error achieved :  0.06818457
'''

# Using Hyperparameter Tuning
def build_model(hp):
    model = keras.Sequential()
    dropout_fraction = hp.Choice('rate', values=[0.1, 0.2, 0.3, 0.4, 0.5])
    model.add(layers.LSTM(units=hp.Int("units" + str(i), min_value=32, max_value=512, step=32),
                          activation=hp.Choice("activation" + str(i), ["relu", "tanh", "sigmoid"]),
                          input_shape=(trainX.shape[1], trainX.shape[2]),
                          return_sequences=False))
    model.add(layers.Dropout(rate=dropout_fraction))
    model.add(layers.Dense(trainY.shape[1]))
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.mean_absolute_error,
    )
    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective=keras_tuner.Objective("val_loss", direction="min"),
    max_trials=10,
    executions_per_trial=3,
    overwrite=True,
    directory="Random_Search",
    project_name="Daily_Revenue_Time_Series",
)

tuner.search(trainX, trainY, epochs=10, validation_data=(testX, testY), verbose=1)
'''
Best val_loss So Far: 0.05872827768325806
Total elapsed time: 00h 03m 24s
'''

# Fetching the best set of parameters
tuner.get_best_hyperparameters()[0].values
'''
{'rate': 0.3, 'units329': 480, 'activation329': 'relu', 'learning_rate': 0.01}
'''

# Fetching the best model
model = tuner.get_best_models(num_models=1)[0]

# Fitting the best model
model.fit(trainX, trainY, epochs=20, batch_size=64, validation_data=(testX, testY), verbose=1)

# Predicting the results with the best model
testPredict = model.predict(testX)
testPredict = np.reshape(testPredict, testY.shape)
print("Mean Absolute Error achieved : ", mean_absolute_error(testY, testPredict))
'''
10/10 [==============================] - 0s 4ms/step
Mean Absolute Error achieved :  0.059796676
'''

# Saving the Model
model.save('daily_revenue_time_series.keras')