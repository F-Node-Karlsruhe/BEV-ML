import matplotlib.pyplot as plt

import data_management

import datetime

import numpy as np

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def plot_data(columns, resample_type, intervall=60):

    intervall = datetime.timedelta(minutes=intervall)

    data_management.loadData()

    if resample_type == 'kwh':
      data_management.DATASET = data_management.DATASET[['delta_kwh']].resample(intervall, label='right', closed='right').sum()
      data_management.DATASET[['delta_kwh']].plot()
      plt.ylabel('kwh charged in ' + str(intervall))

    if resample_type == 'count':
      data_management.DATASET = data_management.DATASET[['time_p']].resample(intervall).count()
      data_management.DATASET[['time_p']].plot(legend=None)
      plt.ylabel('Number of events in ' + str(intervall))

    if resample_type == 'minutes_charged':
      data_management.DATASET = data_management.DATASET[['minutes_charged']].resample(intervall).sum()
      data_management.DATASET[['minutes_charged']].plot(legend=None)
      plt.ylabel('Minutes charged per ' + str(intervall))

    if columns != None:
      data_management.DATASET[columns].plot()
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

def plot_prediction_minutes_charged(data, label, prediction, intervall=60, target=1):

  intervall = datetime.timedelta(minutes=intervall)

  data = data[['minutes_charged']].resample(intervall, label='right', closed='right').sum()

  plot_prediction(data, label, prediction, norm='minutes_charged', y_label='Minutes charged per '+ str(intervall), intervall=intervall, target=target)



def plot_prediction_kwh(data, label, prediction, intervall=60, target=1):

  intervall = datetime.timedelta(minutes=intervall)

  data = data[['delta_kwh']].resample(intervall, label='right', closed='right').sum()

  plot_prediction(data, label, prediction, norm='delta_kwh', y_label='kwh charged per '+ str(intervall) , intervall=intervall, target=target)



def plot_prediction_count(data, label, prediction, intervall=60, target=1):

  intervall = datetime.timedelta(minutes=intervall)

  data = data[['time_p']].resample(intervall, label='right', closed='right').count()

  plot_prediction(data, label, prediction, norm='count', y_label='Number of charges per '+ str(intervall), intervall=intervall, target=target)

  

def plot_prediction(data, label, prediction, norm, y_label, intervall, target):

  eff_error = np.absolute(np.subtract(label, prediction))

  fig=plt.figure()
  ax=fig.add_subplot(111)

  ax.plot(data[1:])

  target_times = [data.index[-1] + intervall * x for x in range(1, target+1)]

  ax.plot(target_times, data_management.denormalizeNumber(label, data_management.NORM_RANGE[norm]), 'rx-', markersize=10,
               label='True Future')
  ax.plot(target_times, data_management.denormalizeNumber(prediction, data_management.NORM_RANGE[norm]), 'go-', markersize=10,
               label='Model Prediction')
  ax.plot(target_times, data_management.denormalizeNumber(eff_error, data_management.NORM_RANGE[norm]), '--c^', markersize=5,
               label='Effective Error')

  plt.title('Prediction example '+y_label)
  plt.ylim(bottom=0)
  plt.legend()
  plt.xlabel('Time')
  plt.ylabel(y_label)
  plt.show()

def plot_error(data, spec, ev_mode):

  fig=plt.figure()
  ax=fig.add_subplot(111)

  for i in range(len(data)):
    ax.plot(data[i], label=spec[i])

  plt.legend()
  plt.ylim(bottom=0)
  plt.title('RSME per ' + ev_mode)
  plt.xlabel(ev_mode)
  plt.ylabel('%')
  plt.show()

def plot_target_error(data, spec):
  plot_error(data, spec, 'target length')

def plot_hour_error(data, spec):
  plot_error(data, spec, 'hour of day')

if __name__ == "__main__":
    print('Visualizer started...')
    plot_data(None, 'kwh')