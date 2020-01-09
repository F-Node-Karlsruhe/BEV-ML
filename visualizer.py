import matplotlib.pyplot as plt

import data_management

import datetime

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def plot_data(columns, resample_type, resample_intervall='1H'):
    data_management.loadData()

    if resample_type == 'kwh':
      data_management.DATASET = data_management.DATASET[['delta_kwh']].resample(resample_intervall, label='right', closed='right').sum()
      data_management.DATASET[['delta_kwh']].plot()
      plt.ylabel('kwh charged in ' + resample_intervall)

    if resample_type == 'count':
      data_management.DATASET = data_management.DATASET[['time_p']].resample(resample_intervall).count()
      data_management.DATASET[['time_p']].plot(legend=None)
      plt.ylabel('Number of events in ' + resample_intervall)

    plt.show()


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.ylabel('error')
    plt.ylabel('epochs')
    plt.legend()

    plt.show()

def plot_prediction_kwh(data, label, prediction, resample_intervall='1H'):

  print(data.tail(10))

  data = data[['delta_kwh']].resample(resample_intervall, label='right', closed='right').sum()

  print(data.tail(5))

  fig=plt.figure()
  ax=fig.add_subplot(111)

  ax.plot(data[1:])

  ax.plot(data.index[-1] + datetime.timedelta(minutes=60), data_management.denormalizeNumber(label, data_management.NORM_RANGE['delta_kwh']), 'rx', markersize=10,
               label='True Future')
  ax.plot(data.index[-1] + datetime.timedelta(minutes=60), data_management.denormalizeNumber(prediction[0][0], data_management.NORM_RANGE['delta_kwh']), 'go', markersize=10,
               label='Model Prediction')

  plt.title('Prediction example kwh')

  plt.legend()
  #plt.yscale('log')
  plt.xlabel('Time')
  plt.ylabel('kwh')
  plt.show()

def plot_prediction_count(plot_data):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']


  plt.title('Prediction example count')
  for i, x in enumerate(plot_data):
    if i:
      plt.plot([len(plot_data[0])+1], plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  #plt.yscale('log')
  plt.xlabel('Time-Step')
  plt.show()

if __name__ == "__main__":
    print('Visualizer started...')
    plot_data(None, 'kwh')