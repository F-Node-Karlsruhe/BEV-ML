import tensorflow as tf 

import data_management

import visualizer

import prediction

import pandas as pd

import os

# Name of the particular model
NAME = 'LSTM'

'''
Prediction settings
'''
# Train model -> False: Predict the prediction timestamp
TRAIN = False

# timestamp till which data is given to predict future (year, month, day, hour)
PREDICTION_TIMESTAMP = pd.Timestamp(2018, 7, 15, 15)


'''
Training parameters
'''
# step size in minutes
STEP_SIZE = 60

# target length in steps in hours
TARGET_LENGTH = int(60/STEP_SIZE) * 96

# history length in hours
HISTORY_LENGTH = STEP_SIZE * int(60/STEP_SIZE) *  96

# label type: ['kwh', 'count', 'minutes_charged']
LABEL_TYPE = 'kwh'

# number of epochs for each training
EPOCHS = 100

# batch size for each step
BATCH_SIZE = 100

# number of steps in each epoch -> 0 = auto
EVALUATION_INTERVAL = 0

# only change if you run into RAM issues
BUFFER_SIZE = 10000

'''
Model parameters
'''
# size of the LSTM output layer
LSTM_SIZE = 2048

# size of the fully connected layer after the LSTM
FULLY_CONNECTED_LAYER_SIZE = LSTM_SIZE * 2

# use a pretained model for training !GPU streamlining not optimal when model loaded!
PRETRAINED = False


def getCallbacks():
    log_dir=data_management.getModelPath(NAME, LSTM_SIZE, LABEL_TYPE, TARGET_LENGTH, STEP_SIZE) #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

# create model specific directory if neccesary
try:
    os.makedirs(data_management.getModelPath(NAME, LSTM_SIZE, LABEL_TYPE, TARGET_LENGTH, STEP_SIZE))
except FileExistsError:
# directory already exists
    pass


# optional: set seed for reproducability
#tf.random.set_seed(12345)

# enable gpu processing on windows10
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

model = None

if TRAIN:
    # init datamanagement
    data_management.init(LABEL_TYPE, STEP_SIZE)

    # get dataset
    x_train, y_train = data_management.getTrainDataset(HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE, STEP_SIZE)
    x_val, y_val = data_management.getValDataset(HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE, STEP_SIZE)

    # set evaluation interval dependent on the dataset size
    if EVALUATION_INTERVAL == 0:
        EVALUATION_INTERVAL = int(len(x_train) / BATCH_SIZE)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.shuffle(1000).batch(BATCH_SIZE).repeat()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(LSTM_SIZE, input_shape=(None, x_train.shape[-1])))
    model.add(tf.keras.layers.Dense(FULLY_CONNECTED_LAYER_SIZE, activation='relu'))
    model.add(tf.keras.layers.Dense(FULLY_CONNECTED_LAYER_SIZE, activation='relu'))
    model.add(tf.keras.layers.Dense(TARGET_LENGTH, activation='relu'))

    model.compile(optimizer='adam', loss='mse')

# load already existing model
if PRETRAINED or not TRAIN:
    print('Loaded ' + data_management.getModelPath(NAME, LSTM_SIZE, LABEL_TYPE, TARGET_LENGTH, STEP_SIZE))
    model = tf.keras.models.load_model(data_management.getModelPath(NAME, LSTM_SIZE, LABEL_TYPE, TARGET_LENGTH, STEP_SIZE))

if TRAIN:
    history = model.fit(train_data, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data,
                                            validation_steps=int(len(x_val) / BATCH_SIZE),
                                            callbacks=getCallbacks())
    # save model
    model.save(data_management.getModelPath(NAME, LSTM_SIZE, LABEL_TYPE, TARGET_LENGTH, STEP_SIZE))
    
    # show train history
    visualizer.plot_train_history(history, NAME + ' ' + LABEL_TYPE + ' ' + str(LSTM_SIZE))

else:
    prediction.predict(model, PREDICTION_TIMESTAMP, HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE, STEP_SIZE)

