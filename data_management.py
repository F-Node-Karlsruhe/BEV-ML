import pandas as pd
import datetime
import numpy as np

DATE_COLUMS = ['time_p', 'time_unp', 'time_fin']

NORM_RANGE = {'c_battery_size_max': (0, 100000), 'c_kombi_current_remaining_range_electric': (0, 500), 'soc_p': (0, 100), 'soc_unp': (0, 100), 'delta_km': (0, 500), 'c_temperature': (-20, 40), 'delta_kwh': (0, 100)}

PATH_PREP = 'data/prep.csv'

DATASET = None


def readRelevantColumns ():
    return pd.read_csv(PATH_PREP, sep=';', parse_dates=DATE_COLUMS)

# loads the DATASET in PATH_PREP
def loadData():
    global DATASET
    DATASET = readRelevantColumns()

# normalized a given number in a given range to [0, 1]
def normalizeNumber(x, bounds):
    space = bounds[1] - bounds[0]
    return (x - bounds[0]) / (space)

def normalizeDatetime(x):
    return x.weekday() / 7

def getNormWeekOfYear(x):
    return x.weekofyear / 52

def normalizeData():
    global DATASET
    # normalize times in week 
    for col in DATE_COLUMS:
        DATASET[col+'_n'] = DATASET[col].apply(normalizeDatetime)
    
    # add week of year
    DATASET['week_of_year'] = DATASET['time_p'].apply(getNormWeekOfYear)

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

def getSample(dataset, start_index, end_index, history_size, target_size):
    """
    Returns the labeled samplein the desired range.
    """
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(DATASET) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)


if __name__ == "__main__":
    loadData()
    normalizeData()
    print(DATASET.head(50))