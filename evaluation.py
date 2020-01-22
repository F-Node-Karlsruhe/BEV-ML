import tensorflow as tf

import pandas as pd

import numpy as np

import data_management

import visualizer

import datetime

'''
Evaluation parameters
'''
# ['basic', 'target', 'hour']
EVALUATION_MODE = 'target'


'''
Model parameters
'''
# name of the model
NAME = 'LSTM'

# label type: ['kwh', 'count', 'minutes_charged']
LABEL_TYPE = 'minutes_charged'

# step size in minutes -> 0 for auto
STEP_SIZE = 60

# size of the memory cell output layer
CELL_SIZE = 1024

# target length in steps in hours
TARGET_LENGTH = int(60/STEP_SIZE) * 24

# history length in hours
HISTORY_LENGTH = STEP_SIZE * int(60/STEP_SIZE) *  48

# load model
model = tf.keras.models.load_model(data_management.getModelPath(NAME, CELL_SIZE, LABEL_TYPE, TARGET_LENGTH, STEP_SIZE))

# load dataset
EVAL_DATA = data_management.getEvaluationData(TARGET_LENGTH, LABEL_TYPE, STEP_SIZE)



# evaluate depending on the mode
if EVALUATION_MODE == 'basic':

    x_eval, y_eval = data_management.getTFDataset(EVAL_DATA, HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE, STEP_SIZE)

    results = model.evaluate(x_eval, y_eval, batch_size=100)

    print(results)

if EVALUATION_MODE == 'target':

    x_eval, y_eval = data_management.getTFDataset(EVAL_DATA, HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE, STEP_SIZE)

    pred = model.predict(x_eval)

    error = np.absolute(np.subtract(pred, y_eval)) * 100

    visualizer.plot_target_error(np.mean(error, axis=0), LABEL_TYPE)

if EVALUATION_MODE == 'hour':

    hour_eval_data = [[] for i in range(24)]

    start_date = EVAL_DATA.index[0] + datetime.timedelta(minutes=HISTORY_LENGTH)

    end_date = EVAL_DATA.index[-1] - datetime.timedelta(minutes=TARGET_LENGTH*STEP_SIZE)

    step = datetime.timedelta(minutes=STEP_SIZE)

    while start_date < end_date:

        features = np.array(EVAL_DATA[start_date-datetime.timedelta(minutes=HISTORY_LENGTH):start_date], ndmin=3)

        label = np.array(data_management.getLabel(EVAL_DATA[start_date:start_date+step*TARGET_LENGTH], LABEL_TYPE, start_date, TARGET_LENGTH, step))

        pred = model.predict(features)[0]

        error = np.absolute(np.subtract(pred, label)) * 100

        for hours in range(len(error)):

            hour = start_date + datetime.timedelta(hours=hours + 1)
            hour = hour.hour

            hour_eval_data[hour].append(error[hours])

        start_date += step
    
    result = np.zeros(24)

    for idx, val in enumerate(hour_eval_data):
        result[idx] = np.mean(val)
     
    print(result)

    visualizer.plot_hour_error(result, LABEL_TYPE)
        
    
    



