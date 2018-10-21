import matplotlib.pyplot as plt


def visualize(history):
    plot_loss(history)
    plot_acc(history)


def plot_loss(history):
    # Get training and test loss histories
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('model loss')
    plt.show()


def plot_acc(history):
    # Get training and test loss histories
    training_acc = history.history['acc']
    test_acc = history.history['val_acc']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_acc) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_acc, 'r--')
    plt.plot(epoch_count, test_acc, 'b-')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('model accuracy')
    plt.show()
