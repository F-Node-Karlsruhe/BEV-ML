import matplotlib.pyplot as plt

import data_management

import datetime

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def plot_data(columns, resample_type, intervall=60):

    intervall = datetime.timedelta(minutes=intervall)

    data_management.loadData()

    if resample_type == 'kwh':
      data_management.DATASET = data_management.DATASET[['delta_kwh']].resample(intervall, label='right', closed='right').sum()
      data_management.DATASET[['delta_kwh']].plot()
      #plt.ylabel('kwh charged in ' + intervall.minutes + ' minutes')

    if resample_type == 'count':
      data_management.DATASET = data_management.DATASET[['time_p']].resample(intervall).count()
      data_management.DATASET[['time_p']].plot(legend=None)
      #plt.ylabel('Number of events in ' + intervall + ' minutes')

    plt.show()


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.ylabel('mse')
    plt.xlabel('epochs')
    plt.legend()

    plt.show()

def plot_prediction_kwh(data, label, prediction, intervall=60, target=1):

  intervall = datetime.timedelta(minutes=intervall)

  data = data[['delta_kwh']].resample(intervall, label='right', closed='right').sum()

  fig=plt.figure()
  ax=fig.add_subplot(111)

  ax.plot(data[1:])

  target_times = [data.index[-1] + intervall * x for x in range(1, target+1)]
 
  ax.plot(target_times, data_management.denormalizeNumber(label, data_management.NORM_RANGE['delta_kwh']), 'rx-', markersize=10,
               label='True Future')
  ax.plot(target_times, data_management.denormalizeNumber(prediction[0], data_management.NORM_RANGE['delta_kwh']), 'go-', markersize=10,
               label='Model Prediction')

  plt.title('Prediction example kwh')

  plt.legend()
  plt.xlabel('Time')
  plt.ylabel('kwh')
  plt.show()

def plot_prediction_count(data, label, prediction, intervall=60, target=1):

  intervall = datetime.timedelta(minutes=intervall)

  data = data[['time_p']].resample(intervall, label='right', closed='right').count()

  fig=plt.figure()
  ax=fig.add_subplot(111)

  ax.plot(data[1:])

  target_times = [data.index[-1] + intervall * x for x in range(1, target+1)]

  ax.plot(target_times, data_management.denormalizeNumber(label, data_management.NORM_RANGE['count']), 'rx', markersize=10,
               label='True Future')
  ax.plot(target_times, data_management.denormalizeNumber(prediction[0], data_management.NORM_RANGE['count']), 'go', markersize=10,
               label='Model Prediction')

  plt.title('Prediction example count')

  plt.legend()
  plt.xlabel('Time')
  plt.ylabel('Number of charges')
  plt.show()

if __name__ == "__main__":
    print('Visualizer started...')
    plot_data(None, 'kwh')