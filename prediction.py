import tensorflow as tf

import pandas as pd

import data_management

import visualizer

'''
Prediciton parameters
'''
# timestamp till which data is given to predict future (year, month, day, hour, minute)
PREDICTION_TIMESTAMP = pd.Timestamp(2018, 12, 3, 6)

# PLZ prediciton
PLZ = None#'7'

'''
Model parameters
'''
# name of the model
NAME = 'LSTM'

# label type: ['kwh', 'count']
LABEL_TYPE = 'kwh'

# size of the memory cell output layer
CELL_SIZE = 512

# history length in minutes
HISTORY_LENGTH = 60 * 24

# target length in steps
TARGET_LENGTH = 8

# step size in minutes -> 0 for auto
STEP_SIZE = 60

def getModelPath():
    '''
        returns the path for the model containing the model specific parameters
    '''
    return 'models/' + NAME + '_' + str(CELL_SIZE) + '__label_' + LABEL_TYPE + '__target_' + str(TARGET_LENGTH) + '__step_' + str(STEP_SIZE)


# enable gpu processing on windows10
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# try to load the specific model
model = tf.keras.models.load_model(getModelPath())

data, norm_data, label = data_management.getTestData(PREDICTION_TIMESTAMP, HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE, STEP_SIZE, PLZ)

prediction = model.predict(norm_data)

print('Prediction: ', prediction[0])

print('True value: ', label)

if LABEL_TYPE == 'kwh':
    visualizer.plot_prediction_kwh(data, label, prediction, intervall=STEP_SIZE, target=TARGET_LENGTH)
if LABEL_TYPE == 'count':
    visualizer.plot_prediction_count(data, label, prediction, intervall=STEP_SIZE, target=TARGET_LENGTH)

#loss,acc = model.evaluate(x_val, y_val, batch_size=100)

#print("Final model, accuracy: {:5.2f}%".format(100*acc))
