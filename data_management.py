import pandas as pd
import datetime
import numpy as np

# reduce if your RAM is not sufficient
DATASET_USAGE = 1.0

TRAIN_SPLIT = 0.8

DATE_COLUMS = ['time_p', 'time_unp', 'time_fin']

# RELEVANT_COLUMNS = ['c_battery_size_max', 'c_kombi_current_remaining_range_electric', 'soc_p', 'soc_unp', 'delta_km', 'c_temperature', 'delta_kwh']

NORM_RANGE = {'c_battery_size_max': (0, 100000), 'c_kombi_current_remaining_range_electric': (0, 500), 'soc_p': (0, 100), 'soc_unp': (0, 100), 'delta_km': (0, 500), 'c_temperature': (-20, 40), 'delta_kwh': (0, 100)}

PATH_PREP = 'data/prep_test.csv'

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

def normalizeDatetime(x):
    return ( x.weekday() * 24 * 60 + x.hour * 60 + x.minute ) / 7 / 24 / 60  # normalize on minutely bases

def getNormWeekOfYear(x):
    return x.weekofyear / 52

def normalizeData():
    global DATASET
    print('Normalizing data ...')
    # add week of year
    DATASET['week_of_year'] = DATASET['time_p'].apply(getNormWeekOfYear)
    # normalize times in week 
    for col in DATE_COLUMS:
        DATASET[col] = DATASET[col].apply(normalizeDatetime)

    # normalize temperature [-20, +40]
    DATASET['c_temperature'] = DATASET['c_temperature'].apply(normalizeNumber, args=[NORM_RANGE['c_temperature']])

    # normalize batterysize [0, 100000]
    DATASET['c_battery_size_max'] = DATASET['c_battery_size_max'].apply(normalizeNumber, args=[NORM_RANGE['c_battery_size_max']])
    # normalize kwh & SOC [0, 100]
    DATASET['soc_p'] = DATASET['soc_p'].apply(normalizeNumber, args=[NORM_RANGE['soc_p']])
    DATASET['soc_unp'] = DATASET['soc_unp'].apply(normalizeNumber, args=[NORM_RANGE['soc_unp']])
    DATASET['delta_kwh'] = DATASET['delta_kwh'].apply(normalizeNumber, args=[NORM_RANGE['delta_kwh']])

    # normalize current electric range & delta_km [0, 500]
    DATASET['c_kombi_current_remaining_range_electric'] = DATASET['c_kombi_current_remaining_range_electric'].apply(normalizeNumber, args=[NORM_RANGE['c_kombi_current_remaining_range_electric']])
    DATASET['delta_km'] = DATASET['delta_km'].apply(normalizeNumber, args=[NORM_RANGE['delta_km']])

# sum of loaded kwh plugged after current time
def getKWHLabel(df, current_time):
    sum = 0
    for index, row in df.iterrows():
        if row['time_p'] > current_time:
            sum += row['delta_kwh']
    return sum

# returns the label dependent on the selected label type
def getLabel(df, labelType, current_time):
    current_time = normalizeDatetime(current_time)
    if labelType == 'kwh':
        return getKWHLabel(df, current_time)
    return 0

def getTFDataset(dataset, history, target_time, lable_type, step=0 ):
    data = []
    labels = []

    start_date = dataset.index[0] + datetime.timedelta(hours=history)
    end_date = dataset.index[-1] - datetime.timedelta(hours=target_time)

    if step == 0:
        step = int(history / 5) # set auto step on 1/5 of the history size
        if step == 0:
            step = 1            # at least 1
    
    step = datetime.timedelta(hours=step)

    print('Labeling data ...')

    while start_date < end_date:

        data.append(np.array(dataset[start_date-datetime.timedelta(hours=history):start_date]))

        labels.append(np.array(getLabel(dataset[start_date-datetime.timedelta(hours=history):start_date+datetime.timedelta(hours=target_time)], lable_type, start_date)))

        start_date += step

    return np.array(data), np.array(labels)

# returns a training dataset based on history in hours, target in hours and label type
def getTrainDataset(history, target_time, label_type, step=0):
    global DATASET
    return getTFDataset(DATASET[:TRAIN_SPLIT], history, target_time, label_type, step=step)

# returns a test dataset based on history in hours, target in hours and label type
def getValDataset(history, target_time, label_type, step=0):
    global DATASET
    return getTFDataset(DATASET[TRAIN_SPLIT:], history, target_time, label_type, step=step)

# inits the data management
def init():
    loadData()
    normalizeData()

if __name__ == "__main__":
    loadData()
    normalizeData()