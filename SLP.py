import datetime

import data_management

import numpy as np

# 10 min intervall of load
SLP = np.array([0.074,
0.068,
0.061,
0.054,
0.044,
0.041,
0.038,
0.036,
0.034,
0.032,
0.030,
0.029,
0.027,
0.025,
0.024,
0.022,
0.021,
0.020,
0.020,
0.020,
0.019,
0.017,
0.017,
0.016,
0.016,
0.016,
0.017,
0.019,
0.022,
0.027,
0.034,
0.044,
0.057,
0.067,
0.078,
0.094,
0.110,
0.132,
0.160,
0.187,
0.218,
0.257,
0.293,
0.328,
0.369,
0.397,
0.410,
0.415,
0.410,
0.397,
0.383,
0.366,
0.351,
0.336,
0.318,
0.298,
0.281,
0.266,
0.257,
0.250,
0.248,
0.250,
0.262,
0.269,
0.272,
0.269,
0.260,
0.241,
0.225,
0.214,
0.208,
0.209,
0.207,
0.203,
0.203,
0.205,
0.206,
0.217,
0.230,
0.245,
0.263,
0.276,
0.286,
0.295,
0.295,
0.294,
0.297,
0.296,
0.297,
0.303,
0.304,
0.305,
0.308,
0.318,
0.336,
0.365,
0.390,
0.412,
0.429,
0.437,
0.434,
0.442,
0.447,
0.446,
0.453,
0.459,
0.453,
0.450,
0.444,
0.429,
0.416,
0.404,
0.389,
0.377,
0.363,
0.343,
0.328,
0.312,
0.295,
0.279,
0.261,
0.240,
0.221,
0.203,
0.187,
0.178,
0.173,
0.169,
0.167,
0.165,
0.160,
0.155,
0.147,
0.138,
0.129,
0.119,
0.111,
0.104,
0.099,
0.094,
0.089,
0.087,
0.084,
0.082,
0.078])

MAX = np.amax(SLP)

def predict(timestamp, target=1, step_size=60):
    result = []
    for _ in range(target):
        index = (timestamp.hour + 1 ) * 6 + int(timestamp.minute / 10 )
        result.append(SLP[index]/MAX * 0.6)
        timestamp = timestamp + datetime.timedelta(minutes=step_size)
    return np.array(result)


def evaluate(EVAL_DATA):

    hour_eval_data = [[] for i in range(24)]

    start_date = EVAL_DATA.index[0]

    end_date = EVAL_DATA.index[-1]

    step = datetime.timedelta(minutes=60)

    while start_date < end_date:

        label = np.array(data_management.getLabel(EVAL_DATA[start_date:start_date+step], 'kwh', start_date, 1, step))

        pred = predict(start_date+step)[0]

        error = np.square(np.subtract(pred, label))

        for hours in range(len(error)):

            hour = start_date + datetime.timedelta(hours=hours + 1)

            hour = hour.hour

            hour_eval_data[hour].append(error[hours])

        start_date += step
        
    result = np.zeros(24)

    for idx, val in enumerate(hour_eval_data):
            result[idx] = np.sqrt(np.mean(val)) * 100
        
    print(result)

    return result