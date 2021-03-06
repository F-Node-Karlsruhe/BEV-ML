import pandas as pd
import datetime
import numpy as np
import os
import sys

# dtype
DTYPE = 'float32'

# reduce if your RAM is not sufficient
DATASET_USAGE = 1.0

# train split after resampling
TRAIN_SPLIT = 0.9

TRAIN_SPLIT_EFF = 0

# resample intervall in minutes
RESAMPLE_INTERVALL = 60

DATE_COLUMS = ['time_p', 'time_unp', 'time_fin']

# RELEVANT_COLUMNS = ['c_battery_size_max', 'c_kombi_current_remaining_range_electric', 'soc_p', 'soc_unp', 'delta_km', 'c_temperature', 'delta_kwh']

# default norm ranges
NORM_RANGE = {'c_battery_size_max': (0, 100000), 'c_kombi_current_remaining_range_electric': (0, 500), 'soc_p': (0, 100), 'soc_unp': (0, 100), 'delta_km': (0, 500), 'c_temperature': (-20, 40), 'delta_kwh': (0, 100)}

# dat to the prepared dataset
PATH_PREP = 'data/prep.csv'



DATASET = None

def readPrepData ():
    return pd.read_csv(PATH_PREP, sep=';', parse_dates=DATE_COLUMS)

def getModelPath(name, cell_size, label_type, target_length, step_size):
    '''
        returns the path for the model containing the model specific parameters
    '''
    return os.path.join(
    'models',
    name + '_' + str(cell_size) + '__label_' + label_type + '__target_' + str(target_length) + '__step_' + str(step_size))

# loads the DATASET in PATH_PREP
def loadData():
    global DATASET
    global TRAIN_SPLIT
    print('Reading data ...')
    # read the data file
    DATASET = readPrepData()
    # set time_p index
    DATASET.set_index('time_p', inplace=True, drop=False)
    # reduce dataset if neccessary
    if DATASET_USAGE < 1.0:
        DATASET = DATASET[:int(len(DATASET.index) * DATASET_USAGE)]

def filterPLZ(dataset, PLZ):
    dataset['PLZ'] = dataset['PLZ'].astype(str)
    return dataset[dataset['PLZ'].str[0:len(PLZ)] == PLZ]

# normalized a given number in a given range to [0, 1]
def normalizeNumber(x, bounds):
    space = bounds[1] - bounds[0]
    return (x - bounds[0]) / (space)

def denormalizeNumber(x, bounds):
    space = bounds[1] - bounds[0]
    return bounds[0] + x * space

def normalizeDatetime(x):
    return ( x.hour * 60 + x.minute ) / 24 / 60  # normalize daytime on minutely bases

def getNormWeekOfYear(x):
    return x.weekofyear / 52

def getNormDayOfWeek(x):
    return x.dayofweek / 7

def normalizeData(dataset, label_type, intervall=RESAMPLE_INTERVALL):

    intervall = datetime.timedelta(minutes=intervall)

    global NORM_RANGE

    global TRAIN_SPLIT_EFF

    print('Resample and normalize data ...')
    # init nor data with time
    norm_data = pd.DataFrame(dataset['time_p'].apply(normalizeDatetime).resample(intervall, label='right', closed='right').min())

    # add week of year
    norm_data['week_of_year'] = dataset['time_p'].apply(getNormWeekOfYear).resample(intervall, label='right', closed='right').mean()
    norm_data['day_of_week'] = dataset['time_p'].apply(getNormDayOfWeek).resample(intervall, label='right', closed='right').mean()

    # normalize and resample temperature [-20, +40]
    norm_data['c_temperature'] = dataset['c_temperature'].apply(normalizeNumber, args=[NORM_RANGE['c_temperature']]).resample(intervall, label='right', closed='right').mean()
    
    # normalize times in week 
    # for col in DATE_COLUMS:
        # dataset[col] = dataset[col].apply(normalizeDatetime)
    if label_type == 'kwh' or label_type == 'minutes_charged':

        # normalize and resample batterysize [0, 100000]
        norm_data['c_battery_size_max'] = dataset['c_battery_size_max'].resample(intervall, label='right', closed='right').sum()
        NORM_RANGE['c_battery_size_max'] = (0, norm_data['c_battery_size_max'].max())
        norm_data['c_battery_size_max'] = norm_data['c_battery_size_max'].apply(normalizeNumber, args=[NORM_RANGE['c_battery_size_max']])

        # normalize and resample  SOC [0, 100]
        norm_data['soc_p'] = dataset['soc_p'].apply(normalizeNumber, args=[NORM_RANGE['soc_p']]).resample(intervall, label='right', closed='right').mean()
        norm_data['soc_unp'] = dataset['soc_unp'].apply(normalizeNumber, args=[NORM_RANGE['soc_unp']]).resample(intervall, label='right', closed='right').mean()


    if label_type == 'kwh':
        # normalize and resample delta kwh
        norm_data['delta_kwh'] = dataset['delta_kwh'].resample(intervall, label='right', closed='right').sum()
        NORM_RANGE['delta_kwh'] = (0, norm_data['delta_kwh'].max())
        norm_data['delta_kwh'] = norm_data['delta_kwh'].apply(normalizeNumber, args=[NORM_RANGE['delta_kwh']])
    
    # normalize and resample delta kwh
    if label_type == 'count':
        norm_data['count'] = dataset['time_p'].resample(intervall, label='right', closed='right').count()
        NORM_RANGE['count'] = (0, norm_data['count'].max())
        norm_data['count'] = norm_data['count'].apply(normalizeNumber, args=[NORM_RANGE['count']])

    # normalize minutes charged
    if label_type == 'minutes_charged':
        norm_data['minutes_charged'] = dataset['minutes_charged'].resample(intervall, label='right', closed='right').sum()
        NORM_RANGE['minutes_charged'] = (0, norm_data['minutes_charged'].max())
        norm_data['minutes_charged'] = norm_data['minutes_charged'].apply(normalizeNumber, args=[NORM_RANGE['minutes_charged']])

    # fill all nans
    norm_data.fillna(0, inplace=True)

    # adjust train split
    TRAIN_SPLIT_EFF = int(len(norm_data.index) * TRAIN_SPLIT)

    return norm_data



# sum of loaded kwh plugged after current time
def getKWHLabel(df):
    return df['delta_kwh'].sum()

# counts the the events in the target time frame
def getCountLabel(df):
    return df['count'].sum()

# counts the the events in the target time frame
def getMinutesChargedLabel(df):
    return df['minutes_charged'].sum()

# returns the label dependent on the selected label type
def getLabel(df, labelType, current_time, target, step):
    label = []
    relevant_time = current_time + datetime.timedelta(seconds=1)
    for t in range(target):

        if labelType == 'kwh':
            label.append(getKWHLabel(df[relevant_time + t * step: relevant_time + (t+1) * step]))
        if labelType == 'count':
            label.append(getCountLabel(df[relevant_time + t * step: relevant_time + (t+1) * step]))
        if labelType == 'minutes_charged':
            label.append(getMinutesChargedLabel(df[relevant_time + t * step: relevant_time + (t+1) * step]))

    return label

def getTFDataset(dataset, history, target, lable_type, step=0 ):
    data = []
    labels = []

    start_date = dataset.index[0] + datetime.timedelta(minutes=history)
    end_date = dataset.index[-1] - datetime.timedelta(minutes=target*step)

    if step == 0:
        step = datetime.timedelta(minutes=RESAMPLE_INTERVALL)
    
    step = datetime.timedelta(minutes=step)

    count = 0

    print('Labeling data ...')

    while start_date < end_date:

        data.append(np.array(dataset[start_date-datetime.timedelta(minutes=history):start_date], dtype=DTYPE))

        labels.append(np.array(getLabel(dataset[start_date:start_date+step*target], lable_type, start_date, target, step), dtype=DTYPE))

        count+=1

        sys.stdout.write("\r{0} - {1} {2} time steps labeled -> Label: {3}".format(start_date, count, step, labels[-1][0]))
        sys.stdout.flush()

        start_date += step

    print('\n')

    return np.array(data), np.array(labels)

# returns a training dataset based on history in minutes, target in minutes and label type
def getTrainDataset(history, target_time, label_type, step=0):
    global DATASET
    return getTFDataset(DATASET[:TRAIN_SPLIT_EFF], history, target_time, label_type, step=step)

# returns a test dataset based on history in minutes, target in minutes and label type
def getValDataset(history, target_time, label_type, step=0):
    global DATASET
    return getTFDataset(DATASET[TRAIN_SPLIT_EFF:], history, target_time, label_type, step=step)

def getTestData(timestamp, history, target, label_type, step, PLZ=None):
    global DATASET
    loadData()
    if PLZ != None:
        DATASET = filterPLZ(DATASET, PLZ)
    data = DATASET[timestamp-datetime.timedelta(minutes=history):timestamp].copy()
    DATASET = normalizeData(DATASET, label_type, step)
    norm_data = np.array([np.array(DATASET[timestamp-datetime.timedelta(minutes=history):timestamp].copy(), dtype=DTYPE)])
    label = np.array(getLabel(DATASET[timestamp:timestamp+datetime.timedelta(minutes=target*step)], label_type, timestamp, target, datetime.timedelta(minutes=step)), dtype=DTYPE)

    return data, norm_data, label


def getEvaluationData(target, label_type, step):
    global DATASET
    loadData()
    DATASET = normalizeData(DATASET, label_type, step)[TRAIN_SPLIT_EFF:]
    return DATASET

    

# inits the data management
def init(label_type, resample_intervall=RESAMPLE_INTERVALL):
    global DATASET
    loadData()
    DATASET = normalizeData(DATASET, label_type, resample_intervall)

