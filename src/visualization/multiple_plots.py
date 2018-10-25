import matplotlib.pyplot as plt


def plot_multiple_val_acc(histories, title, save=False, path=""):
    """Produces a plot comparing models test accuracy over epochs"""
    for name, model_data in histories.items():
        val_acc = model_data["history"].history['val_acc']
        epochs = range(1, len(val_acc)+1)
        plt.plot(epochs, val_acc, label=name)

    plt.xticks(epochs)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title(title)
    if save:
        plt.savefig(path, format='svg', dpi=1200)
    plt.show()


def plot_multiple_val_loss(histories, title):
    """Produces a plot comparing models test loss over epochs"""
    for name, model_data in histories.items():
        val_loss = model_data["history"].history['val_loss']
        epochs = len(val_loss)
        plt.plot(range(1, epochs+1), val_loss, label=name)

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title(title)
    plt.show()
