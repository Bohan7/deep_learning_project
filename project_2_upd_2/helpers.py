import torch
import math
import matplotlib.pyplot as plot

# this function generates the training and test data
def generate_dict_set(size):
    if not isinstance(size, int) or size <= 0:
        raise ValueError('Size of training and testing sets must be a positive integer')

    data = torch.empty(size, 2).uniform_(0, 1)
    label = (data - 0.5).pow(2).sum(dim=1).sub(1 / (2 * math.pi)).sign().add(1).div(2).long()

    return data, label


# this function visualizes the classification results, red dots are misclassified dots, blue dots are correctly classified ones
def visualize_pred(data, label, pred, optimizer):
    # visualize the prediction results
    plot.figure(figsize=(10, 10))
    plot.suptitle('Predicted result for ' + str(optimizer))

    # plot correctly and wrongly predicted points
    corr_points = (label == pred).view(len(label))
    wrong_points = (label != pred).view(len(label))
    plot.scatter(data[corr_points].t()[0], data[corr_points].t()[1], c='blue', label='correct')
    plot.scatter(data[wrong_points].t()[0], data[wrong_points].t()[1], c='red', label='wrong')

    # plot the circle
    x = torch.tensor((range(0, 1000))) * (2 * math.sqrt(1 / 2 / math.pi) / 1000) + 0.5 - math.sqrt(1 / 2 / math.pi)
    y1 = -torch.sqrt((1 / 2 / math.pi - (x - 0.5) ** 2)) + 0.5
    y2 = torch.sqrt((1 / 2 / math.pi - (x - 0.5) ** 2)) + 0.5
    plot.fill_between(x, y1, y2, facecolor='green', alpha=0.3)

    plot.legend()
    plot.savefig('Predicted result for ' + str(optimizer))
    plot.show()

# this function plots the training loss
def plot_loss(nb_epochs, **losses):
    # plot the training losses for different algorithms
    plot.figure(figsize=(12, 8))
    plot.suptitle('Training loss')
    for optimizer, loss in losses.items():
        plot.plot(range(nb_epochs), loss, label=optimizer)
        plot.xlabel('Epochs')
        plot.ylabel('Loss')
        plot.yscale('log')
    # plot.legend()
    # plot.savefig('Training losses')
    plot.show()

# this function plots the training loss of all the model structures (for n runs)
def plot_loss_n_run(loss_plot, model_names, optim_names, n_run):
    # reshape the loss
    losses = torch.tensor(loss_plot).permute(1, 0, 2)
    losses = losses.view(3, 3, n_run, -1)
    plot.figure(figsize=(12, 9))
    for optim_idx in range(3):
        for model_idx in range(3):
            plot_value = losses[optim_idx][model_idx]
            values_mean = plot_value.mean(axis=0)
            values_std = plot_value.std(axis=0)
            model_label = model_names[model_idx % 3]
            optim_label = optim_names[optim_idx % 3]
            # plot the avaraged loss over 10 runs
            plot.plot(range(len(values_mean)), values_mean, label=str(model_label) + '+' + str(optim_label))
            # the shadow represents the standard error
            plot.fill_between(range(len(values_mean)), values_mean - values_std, values_mean + values_std, alpha=0.1)
    plot.xlabel('Epoch', fontsize=12)
    plot.ylabel('loss', fontsize=14)
    # use log scale for y axis
    plot.yscale('log')
    plot.title('Comparison of Training losses for different models and optimizers', fontsize=13)
    plot.legend(shadow=True,fontsize='x-large')
    plot.savefig('plot_loss')
    plot.show()

# this function plots the test accuracy of all the model structures (for n runs)
def plot_acc_n_run(acc_plot, model_names, optim_names, n_run):
    # reshape the data
    accs = torch.tensor(acc_plot).permute(1, 0)
    accs = accs.view(3, 3, n_run)
    plot.figure(figsize=(12, 9))
    for optim_idx in range(3):
        for model_idx in range(3):
            plot_value = accs[optim_idx][model_idx]
            values_mean = plot_value.mean(axis=0)
            values_std = plot_value.std(axis=0)
            model_label = model_names[model_idx % 3]
            optim_label = optim_names[optim_idx % 3]
            # plot the averaged accuracy over 10 runs
            plot.bar(4 * model_idx + 0.85 * optim_idx + 5, values_mean, yerr=values_std,
                     label=str(model_label) + '+' + str(optim_label), capsize=4)  # , yerr=[-values_std, values_std]
    plot.xticks([])
    plot.ylabel('Test accuracy', fontsize=14)
    if n_run >= 300:
        plot.ylim(0.9, 1)
        y_major_locator = plot.MultipleLocator(0.01)
        ax = plot.gca()
        ax.yaxis.set_major_locator(y_major_locator)
    plot.title('Comparison of Test Accuracies for different models and optimizers', fontsize=13)
    plot.legend(shadow=True,fontsize='x-large')
    plot.savefig('acc_loss')
    plot.show()


# this function returns the default optimal super parameters (found by cross validation)
def super_parameter():
    super_param_sgd_relu_default = {'lr': 0.001, 'mini_batch_size': 1}
    super_param_adam_relu_default = {'lr': 0.0001, 'b1': 0.9, 'b2': 0.999, 'epislon': 1e-9, 'mini_batch_size': 20}
    super_param_rmsprop_relu_default = {'lr': 0.0001, 'mini_batch_size': 20, 'rho': 0.9, 'epislon': 1e-9}

    super_param_sgd_tanh_default = {'lr': 0.01, 'mini_batch_size': 1}
    super_param_adam_tanh_default = {'lr': 0.0001, 'b1': 0.8, 'b2': 0.999, 'epislon': 1e-9, 'mini_batch_size': 10}
    super_param_rmsprop_tanh_default = {'lr': 0.0001, 'mini_batch_size': 20, 'rho': 0.9, 'epislon': 1e-9}

    super_param_sgd_sigmoid_default = {'lr': 0.01, 'mini_batch_size': 20}
    super_param_adam_sigmoid_default = {'lr': 0.0001, 'b1': 0.8, 'b2': 0.99, 'epislon': 1e-9, 'mini_batch_size': 20}
    super_param_rmsprop_sigmoid_default = {'lr': 0.0001, 'mini_batch_size': 20, 'rho': 0.9, 'epislon': 1e-9}

    super_param_sgd_default = {'relu': super_param_sgd_relu_default, 'tanh': super_param_sgd_tanh_default,
                               'sigmoid': super_param_sgd_sigmoid_default}
    super_param_adam_default = {'relu': super_param_adam_relu_default, 'tanh': super_param_adam_tanh_default,
                                'sigmoid': super_param_adam_sigmoid_default}
    super_param_rmsprop_default = {'relu': super_param_rmsprop_relu_default, 'tanh': super_param_rmsprop_tanh_default,
                                   'sigmoid': super_param_rmsprop_sigmoid_default}

    super_param_default = {'sgd': super_param_sgd_default, 'adam': super_param_adam_default,
                           'rmsprop': super_param_rmsprop_default}

    return super_param_default