import tensorflow as tf

import pandas as pd

import data_management

import visualizer

'''
Prediciton parameters
'''
# timestamp till which data is given to predict future (year, month, day, hour, minute)
PREDICTION_TIMESTAMP = pd.Timestamp(2018, 12, 3, 16)

'''
Model parameters
'''
# name of the model
NAME = 'LSTM'

# label type: ['kwh', 'count']
LABEL_TYPE = 'kwh'

# size of the memory cell output layer
CELL_SIZE = 128

# history length in minutes
HISTORY_LENGTH = 60 * 3

# target length in minutes
TARGET_LENGTH = 60

def getModelPath():
    '''
        returns the path for the model containing the model specific parameters
    '''
    return 'models/' + NAME + '_' + LABEL_TYPE + '_' + str(CELL_SIZE)


# enable gpu processing on windows10
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# try to load the specific model
model = tf.keras.models.load_model(getModelPath())

data, norm_data, label = data_management.getTestData(PREDICTION_TIMESTAMP, HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE)

prediction = model.predict(norm_data)

print('Prediction: ', prediction[0][0])

print('True value: ', label)

if LABEL_TYPE == 'kwh':
    visualizer.plot_prediction_kwh(data, label, prediction, intervall=TARGET_LENGTH)
if LABEL_TYPE == 'count':
    visualizer.plot_prediction_count(data, label, prediction, intervall=TARGET_LENGTH)

#loss,acc = model.evaluate(x_val, y_val, batch_size=100)

#print("Final model, accuracy: {:5.2f}%".format(100*acc))
