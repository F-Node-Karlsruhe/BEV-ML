import matplotlib.pyplot as plt

import data_management

def plot_data(columns, resample_intervall='1H'):
    data_management.loadData()
    data_management.DATASET = data_management.DATASET.resample(resample_intervall).sum()
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
    plt.legend()

    plt.show()

if __name__ == "__main__":
    print('Visualizer started...')
    plot_data(['delta_kwh'])