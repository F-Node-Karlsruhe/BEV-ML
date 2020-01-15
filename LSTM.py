import tensorflow as tf 

import data_management

import visualizer

import pandas as pd

import os

# Name of the particular model
NAME = 'LSTM'

'''
Prediction settings
'''
# Train model -> False: Predict the prediction timestamp
TRAIN = True

# timestamp till which data is given to predict future (year, month, day, hour)
PREDICTION_TIMESTAMP = pd.Timestamp(2018, 12, 3, 2)


'''
Training parameters
'''
# target length in steps
TARGET_LENGTH = 8

# step size in minutes
STEP_SIZE = 60

# history length 
HISTORY_LENGTH = STEP_SIZE * 24

# label type: ['kwh', 'count']
LABEL_TYPE = 'kwh'

# number of epochs for each training
EPOCHS = 20

# number of steps in each epoch
EVALUATION_INTERVAL = 1000

# batch size for each step
BATCH_SIZE = 100

BUFFER_SIZE = 10000

# size of the LSTM output layer
LSTM_SIZE = 1024

# size of the fully connected layer after the LSTM
FULLY_CONNECTED_LAYER_SIZE = LSTM_SIZE * 2

# use a pretained model for training !GPU streamlining not optimal when model loaded!
PRETRAINED = False


def getModelPath():
    '''
        returns the path for the model containing the model specific parameters
    '''
    return os.path.join(
    'models',
    NAME + '_' + str(LSTM_SIZE) + '__label_' + LABEL_TYPE + '__target_' + str(TARGET_LENGTH) + '__step_' + str(STEP_SIZE))


def getCallbacks():
    log_dir=getModelPath() #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

# create model specific directory if neccesary
try:
    os.makedirs(getModelPath())
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

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.shuffle(1000).batch(BATCH_SIZE).repeat()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(LSTM_SIZE, input_shape=(None, x_train.shape[-1])))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(FULLY_CONNECTED_LAYER_SIZE, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(TARGET_LENGTH))

    model.compile(optimizer='adam', loss='mse')

# load already existing model
if PRETRAINED or not TRAIN:
    model = tf.keras.models.load_model(getModelPath())

if TRAIN:
    history = model.fit(train_data, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data,
                                            validation_steps=50,
                                            callbacks=getCallbacks())
    # save model
    model.save(getModelPath())
    
    # show train history
    visualizer.plot_train_history(history, NAME + ' ' + LABEL_TYPE + ' ' + str(LSTM_SIZE))

else:
    data, norm_data, label = data_management.getTestData(PREDICTION_TIMESTAMP, HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE, STEP_SIZE)

    prediction = model.predict(norm_data)

    print('Prediction: ', prediction[0])

    print('True value: ', label)

    if LABEL_TYPE == 'kwh':
        visualizer.plot_prediction_kwh(data, label, prediction, intervall=STEP_SIZE, target=TARGET_LENGTH)
    if LABEL_TYPE == 'count':
        visualizer.plot_prediction_count(data, label, prediction, intervall=STEP_SIZE, target=TARGET_LENGTH)

