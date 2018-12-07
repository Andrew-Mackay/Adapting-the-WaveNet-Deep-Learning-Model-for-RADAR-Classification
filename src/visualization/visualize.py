import matplotlib.pyplot as plt


def plot_train_acc(history, title, save=False, path=""):
    """Plot the training accuracy over epochs"""
    if type(history) is not dict:
        history = history.history
    acc = history['acc']
    epochs = range(1, len(acc)+1)
    plt.plot(epochs, acc)
    # plt.xticks(epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.title(title)
    if save:
        plt.savefig(path, format='svg', dpi=1200)
    plt.show()


def plot_train_loss(history, title, save=False, path=""):
    """Plot the training loss over epochs"""
    if type(history) is not dict:
        history = history.history
    loss = history['loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss)
    # plt.xticks(epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title(title)
    if save:
        plt.savefig(path, format='svg', dpi=1200)
    plt.show()


def plot_evaluation_bar(dictionary, labels, title, x_label, y_label, metric="accuracy", save=False, path=""):
    """Compare test accuracy or loss in the form of a bar chart"""
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
    if type(history) is not dict:
        history = history.history
    training_loss = history['loss']
    test_loss = history['val_loss']

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
    if type(history) is not dict:
        history = history.history
    training_acc = history['acc']
    test_acc = history['val_acc']

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
        if type(value) is not dict:
            value = value.history
        plt.plot(value['acc'])
    plt.legend(histories.keys())
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.title('Model Train Accuracy')
    plt.show()


def test_accuracy_test_loss_vs_epoch(history, title, save=False, path=""):
    """ Two y axis, one for test accuracy, the other for test loss """
    if type(history) is not dict:
        history = history.history
    test_accuracy = history['val_acc']
    test_loss = history["val_loss"]
    epochs = range(1, len(test_accuracy)+1)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Accuracy', color=color)
    ax1.plot(epochs, test_accuracy, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Test Loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, test_loss, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if save:
        plt.savefig(path, format='svg', dpi=1200)
    plt.show()
