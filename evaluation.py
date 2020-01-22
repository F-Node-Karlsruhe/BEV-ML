import tensorflow as tf

import pandas as pd

import numpy as np

import data_management

import visualizer

'''
Evaluation parameters
'''
# ['basic']
EVALUATION_MODE = 'target'


'''
Model parameters
'''
# name of the model
NAME = 'LSTM'

# label type: ['kwh', 'count', 'minutes_charged']
LABEL_TYPE = 'kwh'

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

print(EVAL_DATA.head(5))

if EVALUATION_MODE == 'basic':

    x_eval, y_eval = data_management.getTFDataset(EVAL_DATA, HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE, STEP_SIZE)

    results = model.evaluate(x_eval, y_eval, batch_size=100)

    print(results)

if EVALUATION_MODE == 'target':

    x_eval, y_eval = data_management.getTFDataset(EVAL_DATA, HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE, STEP_SIZE)

    pred = model.predict(x_eval)

    error = np.subtract(pred, y_eval)

    visualizer.plot_target_error(np.mean(error, axis=0), LABEL_TYPE)



