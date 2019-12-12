import tensorflow as tf 

import data_management

# Training Parameters
HISTORY_LENGTH = 3

TARGET_LENGTH = 1

LABEL_TYPE = 'kwh'

BATCH_SIZE = 100

BUFFER_SIZE = 10000


# init datamanagement
data_management.init()

# optional: set seed for reproducability
#tf.random.set_seed(12345)

x_train, y_train = data_management.getTrainDataset(HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE)

print(x_train.dtype)
print(y_train.dtype)
print ('Single window of past history : {}'.format(x_train[0][0][0].dtype))


x_val, y_val = data_management.getValDataset(HISTORY_LENGTH, TARGET_LENGTH, LABEL_TYPE)

train_data_single = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

print(len(x_train))