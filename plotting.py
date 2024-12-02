import matplotlib.pyplot as plt


# plot the learning curve, accuracy vs iteration, loss vs iteration
def plotting(data, title='Plot', x_label='iteration', y_label='Y', legend_labels: list = None):
    plt.figure(figsize=(10, 6))
    train = data[0]
    valid = data[1]

    plt.plot(list(train.keys()), list(train.values()), marker='o', linestyle='-')
    plt.plot(list(valid.keys()), list(valid.values()), marker='o', linestyle='-')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()


def plot_accuracy_loss(accuracy, loss, title='accuracy vs loss', x_label='iteration', y_label='Y', legend_labels: list = None):
    plt.figure(figsize=(10, 6))

    plt.plot(list(accuracy.keys()), list(accuracy.values()), marker='o', linestyle='-', label='accuracy')
    plt.plot(list(loss.keys()), list(loss.values()), marker='o', linestyle='-', label='loss')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()


def plot_all(accuracy, loss, title='Plot', x_label='iteration', y_label='Y', legend_labels: list = None):
    if legend_labels is None:
        legend_labels = ['train', 'valid']
    accuracy_train = accuracy[0]
    accuracy_valid = accuracy[1]
    loss_train = loss[0]
    loss_valid = loss[1]
    fig, axs = plt.subplots(3, figsize=(10, 8))
    fig.suptitle(title)
    axs[0].plot(list(accuracy_train.keys()), list(accuracy_train.values()), marker='o', linestyle='-')
    # legend labels for train accuracy
    axs[0].plot(list(accuracy_valid.keys()), list(accuracy_valid.values()), marker='o', linestyle='-')
    # legend labels for valid accuracy
    if legend_labels is not None:
        axs[0].legend(legend_labels)
    axs[0].set_title('Accuracy vs iteration')
    for key, value in accuracy_train.items():
        axs[0].annotate(f'{value:.2f}', (key, value), textcoords="offset points", xytext=(0, 5), ha='center')
    plt.grid(True)
    max_accuracy_key, max_accuracy_value = max(accuracy_train.items(), key=lambda item: item[1])
    axs[0].annotate(f'{max_accuracy_value:.3f}', (max_accuracy_key, max_accuracy_value),
                    textcoords="offset points", xytext=(0, -20), ha='center')

    axs[1].plot(list(loss_train.keys()), list(loss_train.values()), marker='o', linestyle='-')
    axs[1].plot(list(loss_valid.keys()), list(loss_valid.values()), marker='o', linestyle='-')
    axs[1].set_title('Loss vs iteration')
    plt.grid(True)
    if legend_labels is not None:
        axs[1].legend(legend_labels)

    for key, value in loss_train.items():
        axs[1].annotate(f'{value:.2f}', (key, value), textcoords="offset points", xytext=(0, 5), ha='center')
    max_loss_key, max_loss_value = max(loss_valid.items(), key=lambda item: item[1])
    axs[1].annotate(f'{max_loss_value:.3f}', (max_loss_key, max_loss_value),
                    textcoords="offset points", xytext=(0, -20), ha='center')

    axs[2].plot(list(accuracy_train.keys()), list(loss_train.values()), marker='o', linestyle='-')
    axs[2].plot(list(accuracy_valid.keys()), list(loss_valid.values()), marker='o', linestyle='-')
    axs[2].set_title('Accuracy vs Loss')
    if legend_labels is not None:
        axs[2].legend(legend_labels)

    plt.tight_layout(pad=2.0)
    plt.grid(True)
    plt.show()

