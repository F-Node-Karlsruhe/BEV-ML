import pandas as pd

DATE_COLUMS = ['time_p', 'time_unp', 'time_fin']

PATH_RAW = 'data/prep.csv'

DATASET = None


def readRelevantColumns ():
    return pd.read_csv(PATH_RAW, sep=';', parse_dates=DATE_COLUMS)

def loadData():
    global DATASET
    DATASET = readRelevantColumns()


def add_delta_kwh(df):
    df['delta_kwh'] = (df['soc_unp'] - df['soc_p']) * df['c_battery_size_max'] / 100

def getSample(dataset, start_index, end_index, history_size, target_size):
    """
    Returns the labeled samplein the desired range.
    """
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)


if __name__ == "__main__":
    loadData()
    add_delta_kwh(DATASET)
    print(DATASET.head(50))