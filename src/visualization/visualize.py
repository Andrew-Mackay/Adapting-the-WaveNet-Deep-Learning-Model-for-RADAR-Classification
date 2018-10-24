import matplotlib.pyplot as plt


def plot_train_acc(history, title, save=False, path=""):
    acc = history.history['acc']
    epochs = range(1, len(acc)+1)
    plt.plot(epochs, acc)
    plt.xticks(epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.title(title)
    if save:
        plt.savefig(path, format='svg', dpi=1200)
    plt.show()


def plot_train_loss(history, title, save=False, path=""):
    loss = history.history['loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss)
    plt.xticks(epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title(title)
    if save:
        plt.savefig(path, format='svg', dpi=1200)
    plt.show()


def plot_evaluation_bar(dictionary, labels, title, x_label, y_label, metric="accuracy", save=False, path=""):
    heights = []
    for key, value in dictionary.items():
        heights.append(value[metric])
    plt.bar(range(len(dictionary.keys())), heights)
    plt.xticks(range(len(labels)), labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if save:
        plt.savefig(path, format='svg', dpi=1200)
    plt.show()


def visualize(history):
    plot_train_test_loss(history)
    plot_train_test_acc(history)


def plot_train_test_loss(history):
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


def plot_train_test_acc(history):
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


def plot_all_train_acc(histories):
    for key, value in histories.items():
        plt.plot(value.history['acc'])
    plt.legend(histories.keys())
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.title('Model Train Accuracy')
    plt.show()
