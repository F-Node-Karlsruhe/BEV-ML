import matplotlib.pyplot as plt

import data_management

def plot_data(columns, resample_type, resample_intervall='1H'):
    data_management.loadData()

    if resample_type == 'kwh':
      data_management.DATASET = data_management.DATASET[['delta_kwh']].resample(resample_intervall).sum()
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
    plt.legend()

    plt.show()

def plot_prediction_kwh(plot_data, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']


  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot([len(plot_data[0])+1], plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.yscale('log')
  plt.xlabel('Time-Step')
  plt.ylabel('100 kwh')
  plt.show()

if __name__ == "__main__":
    print('Visualizer started...')
    plot_data(None, 'count')