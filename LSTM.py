import tensorflow as tf 

import data_management

'''
Training Parameters
'''
# history length in minutes
HISTORY_LENGTH = 180

# target length in minutes
TARGET_LENGTH = 60

# label type ['kwh']
LABEL_TYPE = 'kwh'

BATCH_SIZE = 100

BUFFER_SIZE = 10000

EPOCHS = 10

EVALUATION_INTERVAL = 200


# init datamanagement
data_management.init()

# optional: set seed for reproducability
#tf.random.set_seed(12345)

# enable gpu processing on windows10
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

x_train, y_train = data_management.getTrainDataset(HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE)
x_val, y_val = data_management.getValDataset(HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE)

# padd the sequences
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, dtype='float32')
x_val = tf.keras.preprocessing.sequence.pad_sequences(x_val, dtype='float32')   

print(x_train.shape[-2:])

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(BATCH_SIZE).repeat()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(32, input_shape=x_train.shape[-2:]))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

history = model.fit(train_data, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data,
                                            validation_steps=50)

