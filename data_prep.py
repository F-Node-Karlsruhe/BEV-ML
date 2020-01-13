import pandas as pd

RELEVANT_COLUMS = {'c_battery_size_max': 'int32', 'c_kombi_current_remaining_range_electric': 'int16', 'time_p': 'str', 'time_unp': 'str', 'time_fin': 'str', 'soc_p': 'float32', 'soc_unp': 'float32', 'delta_km': 'int16', 'c_temperature': 'float32', 'PLZ': 'str'}

DATE_COLUMS = ['time_p', 'time_unp', 'time_fin']

PATH_RAW = 'data/raw.csv'

DATASET = None

def readRelevantColumns ():
    return pd.read_csv(PATH_RAW, sep=';', dtype=RELEVANT_COLUMS, parse_dates=DATE_COLUMS)[RELEVANT_COLUMS.keys()]

def loadData():
    global DATASET
    print('Loading data in memory ...')
    DATASET = readRelevantColumns()
    # set plugin timestamp as index
    print('Indexing ...')
    DATASET.index = DATASET['time_p']
    # sort by index
    print('Sorting dataset ...')
    DATASET.sort_index(inplace=True)

def add_delta_kwh(df):
    df['delta_kwh'] = (df['soc_unp'] - df['soc_p']) * df['c_battery_size_max'] / 100000

if __name__ == "__main__":
    loadData()
    # add features
    print('Adding delta kwh ...')
    add_delta_kwh(DATASET)

    print(DATASET.head(50))
    print('Saving dataset ...')
    DATASET.to_csv('data/prep.csv', sep=';', index=False)