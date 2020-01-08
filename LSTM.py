import tensorflow as tf 

import data_management

import visualizer

import pandas as pd

import os

NAME = 'LSTM'


'''
Training parameters
'''
# history length in minutes
HISTORY_LENGTH = 180

# target length in minutes
TARGET_LENGTH = 60

# label type ['kwh', 'count']
LABEL_TYPE = 'kwh'

BATCH_SIZE = 100

BUFFER_SIZE = 10000

EPOCHS = 100

EVALUATION_INTERVAL = 2000

LSTM_SIZE = 64

PRETRAINED = False

TRAIN = False

def getModelPath():
    global LSTM_SIZE
    global NAME
    global LABEL_TYPE
    return 'models/' + NAME + '_' + LABEL_TYPE + '_' + str(LSTM_SIZE)



# optional: set seed for reproducability
#tf.random.set_seed(12345)

# enable gpu processing on windows10
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

model = None

if TRAIN:
    # init datamanagement
    data_management.init()

    # get dataset
    x_train, y_train = data_management.getTrainDataset(HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE)
    x_val, y_val = data_management.getValDataset(HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE)

    # padd the sequences
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, dtype='float32')
    x_val = tf.keras.preprocessing.sequence.pad_sequences(x_val, dtype='float32')   

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.shuffle(1000).batch(BATCH_SIZE).repeat()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(LSTM_SIZE, input_shape=(None, x_train.shape[-1])))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mae')

# load already existing model
if PRETRAINED or not TRAIN:
    model = tf.keras.models.load_model(getModelPath())

if TRAIN:
    history = model.fit(train_data, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data,
                                            validation_steps=50)

    # save the model to /models !NAME folder must already exist
    try:
        os.makedirs(getModelPath())
    except FileExistsError:
    # directory already exists
        pass
    model.save(getModelPath())
    
    # show trian history
    visualizer.plot_train_history(history, NAME + ' ' + LABEL_TYPE + ' ' + LSTM_SIZE)

else:
    data, norm_data, label = data_management.getTestData(pd.Timestamp(2018, 6, 6, 12), HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE)

    prediction = model.predict(norm_data)

    if LABEL_TYPE == 'kwh':
        visualizer.plot_prediction_kwh(data, label, prediction)

    '''val_data.shuffle(1000)
    for x, y in val_data.take(3):
        for i in range(BATCH_SIZE):
            # only take non zero examples
            if y[i] > 0:
                print(x.shape)
                if LABEL_TYPE == 'kwh':
                    visualizer.plot_prediction_kwh([x[i][:, -2].numpy(), y[i].numpy(),
                            model.predict(x)[i]])
                if LABEL_TYPE == 'count':
                    visualizer.plot_prediction_count([x[i][:, -2].numpy(), y[i].numpy(),
                            model.predict(x)[i]])
                
                
                break'''
