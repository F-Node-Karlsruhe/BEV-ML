import pandas as pd
import datetime
import numpy as np
import sys

# dtype
DTYPE = 'float32'

# reduce if your RAM is not sufficient
DATASET_USAGE = 1.0

TRAIN_SPLIT = 0.9

DATE_COLUMS = ['time_p', 'time_unp', 'time_fin']

# RELEVANT_COLUMNS = ['c_battery_size_max', 'c_kombi_current_remaining_range_electric', 'soc_p', 'soc_unp', 'delta_km', 'c_temperature', 'delta_kwh']

NORM_RANGE = {'c_battery_size_max': (0, 100000), 'c_kombi_current_remaining_range_electric': (0, 500), 'soc_p': (0, 100), 'soc_unp': (0, 100), 'delta_km': (0, 500), 'c_temperature': (-20, 40), 'delta_kwh': (0, 100)}

PATH_PREP = 'data/prep.csv'

DATASET = None


def readPrepData ():
    return pd.read_csv(PATH_PREP, sep=';', parse_dates=DATE_COLUMS)

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
    # determine effective size of train split
    TRAIN_SPLIT = int(len(DATASET.index) * TRAIN_SPLIT)

# normalized a given number in a given range to [0, 1]
def normalizeNumber(x, bounds):
    space = bounds[1] - bounds[0]
    return (x - bounds[0]) / (space)

def denormalizeNumber(x, bounds):
    space = bounds[1] - bounds[0]
    return bounds[0] + x * space

def normalizeDatetime(x):
    return ( x.weekday() * 24 * 60 + x.hour * 60 + x.minute ) / 7 / 24 / 60  # normalize on minutely bases

def getNormWeekOfYear(x):
    return x.weekofyear / 52

def normalizeData(dataset):
    print('Normalizing data ...')
    # add week of year
    dataset['week_of_year'] = dataset['time_p'].apply(getNormWeekOfYear)
    # normalize times in week 
    for col in DATE_COLUMS:
        dataset[col] = dataset[col].apply(normalizeDatetime)

    # normalize temperature [-20, +40]
    dataset['c_temperature'] = dataset['c_temperature'].apply(normalizeNumber, args=[NORM_RANGE['c_temperature']])

    # normalize batterysize [0, 100000]
    dataset['c_battery_size_max'] = dataset['c_battery_size_max'].apply(normalizeNumber, args=[NORM_RANGE['c_battery_size_max']])
    # normalize kwh & SOC [0, 100]
    dataset['soc_p'] = dataset['soc_p'].apply(normalizeNumber, args=[NORM_RANGE['soc_p']])
    dataset['soc_unp'] = dataset['soc_unp'].apply(normalizeNumber, args=[NORM_RANGE['soc_unp']])
    dataset['delta_kwh'] = dataset['delta_kwh'].apply(normalizeNumber, args=[NORM_RANGE['delta_kwh']])

    # normalize current electric range & delta_km [0, 500]
    dataset['c_kombi_current_remaining_range_electric'] = dataset['c_kombi_current_remaining_range_electric'].apply(normalizeNumber, args=[NORM_RANGE['c_kombi_current_remaining_range_electric']])
    dataset['delta_km'] = dataset['delta_km'].apply(normalizeNumber, args=[NORM_RANGE['delta_km']])

# sum of loaded kwh plugged after current time
def getKWHLabel(df, current_time):
    return df[current_time:]['delta_kwh'].sum()

# counts the the events in the target time frame
def getCountLabel(df, current_time):
    return len(df[current_time:].index)

# returns the label dependent on the selected label type
def getLabel(df, labelType, current_time):
    # current_time = normalizeDatetime(current_time)
    if labelType == 'kwh':
        return getKWHLabel(df, current_time)
    if labelType == 'count':
        return getCountLabel(df, current_time)
    return 0

def getTFDataset(dataset, history, target_time, lable_type, step=0 ):
    data = []
    labels = []

    start_date = dataset.index[0] + datetime.timedelta(minutes=history)
    end_date = dataset.index[-1] - datetime.timedelta(minutes=target_time)

    if step == 0:
        step = int(history / 6) # set auto step on 1/6 of the history size
        if step == 0:
            step = 1            # at least 1
    
    step = datetime.timedelta(minutes=step)

    count = 0

    print('Labeling data ...')

    while start_date < end_date:

        data.append(np.array(dataset[start_date-datetime.timedelta(minutes=history):start_date], dtype=DTYPE))

        labels.append(np.array(getLabel(dataset[start_date-datetime.timedelta(minutes=history):start_date+datetime.timedelta(minutes=target_time)], lable_type, start_date), dtype=DTYPE))

        start_date += step

        count+=1

        sys.stdout.write("\r{0} - {1} {2} minute steps labeled -> Label: {3}".format(start_date, count, step, labels[-1]))
        sys.stdout.flush()
    print('\n')

    return np.array(data), np.array(labels)

# returns a training dataset based on history in minutes, target in minutes and label type
def getTrainDataset(history, target_time, label_type, step=0):
    global DATASET
    return getTFDataset(DATASET[:TRAIN_SPLIT], history, target_time, label_type, step=step)

# returns a test dataset based on history in minutes, target in minutes and label type
def getValDataset(history, target_time, label_type, step=0):
    global DATASET
    return getTFDataset(DATASET[TRAIN_SPLIT:], history, target_time, label_type, step=step)

def getTestData(timestamp, history, target_time, label_type):
    global DATASET
    loadData()
    data = DATASET[timestamp-datetime.timedelta(minutes=history):timestamp].copy()
    normalizeData(DATASET)
    norm_data = np.array([np.array(DATASET[timestamp-datetime.timedelta(minutes=history):timestamp].copy(), dtype=DTYPE)])
    label = np.array(getLabel(DATASET[timestamp-datetime.timedelta(minutes=history):timestamp+datetime.timedelta(minutes=target_time)], label_type, timestamp), dtype=DTYPE)

    return data, norm_data, label


# inits the data management
def init():
    global DATASET
    loadData()
    normalizeData(DATASET)

