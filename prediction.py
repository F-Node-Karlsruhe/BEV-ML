import tensorflow as tf

import pandas as pd

import data_management

import visualizer

import SLP

'''
Prediciton parameters
'''
# timestamp till which data is given to predict future (year, month, day, hour, minute)
PREDICTION_TIMESTAMP = pd.Timestamp(2018, 7, 11, 15)

# compare with SLP
PRED_SLP = True

# PLZ prediciton -> set None if not wanted
PLZ = None#'7'

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
HISTORY_LENGTH = STEP_SIZE * int(60/STEP_SIZE) *  96


# enable gpu processing on windows10
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)


def predict(model, time=PREDICTION_TIMESTAMP, history_length=HISTORY_LENGTH, target_length=TARGET_LENGTH, label_type=LABEL_TYPE, step_size=STEP_SIZE, plz=PLZ):

    data, norm_data, label = data_management.getTestData(time, history_length, target_length, label_type, step_size, plz)

    prediction = model.predict(norm_data)[0]

    if PRED_SLP:
        prediction = SLP.predict(PREDICTION_TIMESTAMP, TARGET_LENGTH, step_size)

    print('Prediction: ', prediction)

    print('True value: ', label)

    if label_type == 'kwh':
        visualizer.plot_prediction_kwh(data, label, prediction, intervall=step_size, target=target_length)
    if label_type == 'count':
        visualizer.plot_prediction_count(data, label, prediction, intervall=step_size, target=target_length)
    if label_type == 'minutes_charged':
        visualizer.plot_prediction_minutes_charged(data, label, prediction, intervall=step_size, target=target_length)

    #loss,acc = model.evaluate(x_val, y_val, batch_size=100)

    #print("Final model, accuracy: {:5.2f}%".format(100*acc))


if __name__ == "__main__":
    # try to load the specific model
    model = tf.keras.models.load_model(data_management.getModelPath(NAME, CELL_SIZE, LABEL_TYPE, TARGET_LENGTH, STEP_SIZE))
    predict(model, PREDICTION_TIMESTAMP)
    #while(True):
        #user_input = list(map(int, input("Enter timestamp like 'month day hour':").split()))
        #timestamp = pd.Timestamp(2018, user_input[0], user_input[1], user_input[2])
        #predict(model, timestamp)
