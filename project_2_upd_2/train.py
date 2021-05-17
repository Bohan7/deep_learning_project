# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:22:21 2021

@author: Bohan
"""
import torch
import math
import matplotlib.pyplot as plot
from model import Linear, Sequential, ReLU, Tanh, Loss_MSE, Loss_Cross_Entropy # , Batch_norm

def generate_dict_set(size):
    if not isinstance(size, int) or size <= 0:
        raise ValueError('Size of training and testing sets must be a positive integer')
    
    data = torch.empty(size, 2).uniform_(0, 1)
    label = (data - 0.5).pow(2).sum(dim=1).sub(1 / (2 * math.pi)).sign().add(1).div(2).long()
    
    return data, label
    
def train(model, train_input, train_target, loss, nb_epochs, print_info, super_param, optimizer, early_stopping):
    criterion = loss
    lr = super_param['lr']
    mini_batch_size = super_param['mini_batch_size']
    if optimizer == 'adam':
        b1 = super_param['b1']
        b2 = super_param['b2']
        epislon = super_param['epislon']

    training_loss = []
    # acc_val = [] # used for early stopping
    for e in range(nb_epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            model.zero_grad()

            output = model.forward(train_input.narrow(0, b, mini_batch_size).t())
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size).t())
            acc_loss = acc_loss + loss

            grad_input = criterion.backward()
            model.backward(grad_input)
            if optimizer == 'sgd':
                model.update(lr)
            elif optimizer == 'adam':
                model.update_adam(lr, b1, b2, epislon)
            elif optimizer == 'rmsprop':
                model.update_rmsprop(lr, rho=0.9, epislon=1e-10)
        # early stop if the training loss has not improved for the last 10 iterations
        if early_stopping == True:
            # acc_val.append(test(model, train_input, train_target, plot_result='False', optimizer=optimizer))
            if len(training_loss) >= 30:
                if (training_loss[-1]/training_loss[-11]) >= 0.99:
                    print("Accuracy hasn't improved in 10 iterations, early stop.")
                    print('epoch {}: loss: {}'.format(e, acc_loss))
                    break
        training_loss.append(acc_loss)
        # print training loss every 10 iterations
        if (print_info == True) and ((e%10 == 0) or (e == nb_epochs-1)):
            print('epoch {}: loss: {}'.format(e, acc_loss))

    return training_loss

def test(model, test_input, test_target, plot_result, optimizer):
    prediction = model.forward(test_input.t())
    model.zero_grad()

    predicted_class = (prediction > 0.5).type(torch.FloatTensor)
    accuracy = (predicted_class == test_target).float().mean()

    # plot the visulization of final prediction
    if plot_result == True:
        visualize_pred(test_input, test_target, predicted_class, optimizer)

    return accuracy


def cross_validate(model, train_input, train_target, loss, nb_epochs, sup_param_range, optimizer):
    # retrieve the range of the superparameters from the dictionary
    if optimizer == 'sgd':
        lr_range = sup_param_range['lr_range']
        mini_batch_size_range = sup_param_range['mini_batch_size_range']
        params_range = [(lr, mini_batch_size) for lr in lr_range for mini_batch_size in mini_batch_size_range]
    elif optimizer == 'adam':
        lr_range = sup_param_range['lr_range']
        mini_batch_size_range = sup_param_range['mini_batch_size_range']
        b1_range = sup_param_range['b1_range']
        b2_range = sup_param_range['b2_range']
        epislon_range = sup_param_range['epislon_range']
        params_range = [(lr, b1, b2, epislon, mini_batch_size) for lr in lr_range for b1 in b1_range \
                        for b2 in b2_range for epislon in epislon_range for mini_batch_size in mini_batch_size_range]
    elif optimizer == 'rmsprop':
        lr_range = sup_param_range['lr_range']
        mini_batch_size_range = sup_param_range['mini_batch_size_range']
        rho_range = sup_param_range['rho_range']
        epislon_range = sup_param_range['epislon_range']
        params_range = [(lr, mini_batch_size, rho, epislon) for lr in lr_range for mini_batch_size in mini_batch_size_range \
                        for rho in rho_range for epislon in epislon_range]
    # use 5-fold cross validation
    n_fold = 5
    fold_size = int(len(train_input)/n_fold)
    val_acc_mean = []
    print("Cross validation starts for " + str(optimizer))
    # iterate over the parameters to find the optimal superparameters
    for params in params_range:
        val_acc = []
        for i in range(n_fold):
            model.init_params()
            model.zero_grad()

            # split to training and validation set
            train_set = torch.cat((train_input[0:i*fold_size, :], train_input[(i+1)*fold_size: len(train_input):, :]), 0)
            train_set_target = torch.cat((train_target[0: i*fold_size], train_target[(i+1)*fold_size: len(train_target)]), 0)

            val_set = train_input[i*fold_size : (i+1)*fold_size, :]
            val_set_target = train_target[i*fold_size : (i+1)*fold_size]

            if optimizer == 'sgd':
                super_param = {'lr': params[0], 'mini_batch_size': params[1]}
            elif optimizer == 'adam':
                super_param = {'lr': params[0], 'b1': params[1], 'b2': params[2], 'epislon': params[3], 'mini_batch_size': params[4]}
            elif optimizer == 'rmsprop':
                super_param = {'lr': params[0], 'mini_batch_size': params[1], 'rho': params[2], 'epislon': params[3]}

            train(model, train_set, train_set_target, loss, 500, False, super_param, optimizer, False) # nb_epochs暂时设为2
            val_acc.append(test(model, val_set, val_set_target, False, optimizer))
        val_acc_mean.append(sum(val_acc)/n_fold)

        # print validation result for each combination of superparameters
        if optimizer == 'sgd':
            print("for lr: {}, mini_batch_size : {}, validation accuracy is {}".format(params[0], params[1], float(sum(val_acc) / n_fold)))
        elif optimizer == 'adam':
            print("for lr: {}, b1: {}, b2: {}, epislon: {}, mini_batch_size : {}, validation accuracy is {}"\
                  .format(params[0],params[1],params[2],params[3],params[4], float(sum(val_acc) / n_fold)))
        elif optimizer == 'rmsprop':
            print("for lr: {}, mini_batch_size: {}, rho: {}, epislon: {}, validation accuracy is {}"\
                  .format(params[0],params[1],params[2],params[3], float(sum(val_acc) / n_fold)))

    # find the optimal parameters (with highest average validation accuracy)
    argmax_acc = [i for i, val in enumerate(val_acc_mean) if (val == max(val_acc_mean))][0]
    opt_param = params_range[argmax_acc]
    # print the optimal parameters found by cross validation
    if optimizer == 'sgd':
        print("The optimal params for sgd are, lr: {}, mini_batch_size: {}"\
              .format(opt_param[0],opt_param[1]))
    elif optimizer == 'adam':
        print("The optimal params for adam are, lr: {}, b1: {}, b2:{}, epislon: {}, mini_batch_size: {}"\
              .format(opt_param[0],opt_param[1],opt_param[2],opt_param[3], opt_param[4]))
    elif optimizer == 'rmsprop':
        print("The optimal params for rmsprop are, lr: {}, mini_batch_size: {}, rho:{}, epislon: {}"\
              .format(opt_param[0],opt_param[1],opt_param[2],opt_param[3]))
    return opt_param


def plot_loss(nb_epochs, **losses):
    # plot the training losses for different algorithms
    plot.figure(figsize=(12,8))
    plot.suptitle('Training losses for different optimizers')
    for optimizer, loss in losses.items():
        plot.plot(range(nb_epochs), loss, label = optimizer)
        plot.xlabel('Epochs')
        plot.ylabel('Loss')
        plot.yscale('log')
    plot.legend()
    plot.savefig('Training losses')
    plot.show()

def visualize_pred(data, label, pred, optimizer):
    # visualize the prediction results
    plot.figure(figsize=(10,10))
    plot.suptitle('Predicted result for ' + str(optimizer))

    # plot correctly and wrongly predicted points
    corr_points = (label==pred).view(len(label))
    wrong_points = (label!=pred).view(len(label))
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

def main():
    train_input, train_target = generate_dict_set(1000)
    test_input, test_target = generate_dict_set(1000)

    relu_model = Sequential(Linear(2, 25), ReLU(),
                            Linear(25, 25), ReLU(),
                            Linear(25, 25), ReLU(),
                            Linear(25, 1), init_type='xavier', init_gain=1.0, epsilon=1e-6)

    """
    # do cross-validation to find super parameters
    sup_param_cv_range_sgd = {'lr_range' : [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'mini_batch_size_range':[1]}
    opt_param_sgd = cross_validate(relu_model, train_input, train_target, Loss_MSE(), 300, sup_param_cv_range_sgd, 'sgd')

    sup_param_cv_range_adam = {'lr_range' : [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                               'b1_range': [0.9, 0.8],
                               'b2_range': [0.999, 0.99],
                               'epislon_range': [1e-8, 1e-9],
                               'mini_batch_size_range':[10,20]}
    opt_param_adam = cross_validate(relu_model, train_input, train_target, Loss_MSE(), 300, sup_param_cv_range_adam, 'adam')

    sup_param_cv_range_rmsprop = {'lr_range' : [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                                  'mini_batch_size_range':[10,20],
                                  'rho_range': [0.9, 0.8],
                                  'epislon_range': [1e-8, 1e-9]}
    opt_param_rmsprop = cross_validate(relu_model, train_input, train_target, Loss_MSE(), 300, sup_param_cv_range_rmsprop, 'rmsprop')

    # set the superparameters to the optimal superparameters we found in cross validation
    super_param_sgd = {'lr': opt_param_sgd[0], 'mini_batch_size': opt_param_sgd[1]}
    super_param_adam = {'lr': opt_param_adam[0], 'b1': opt_param_adam[1], 'b2':opt_param_adam[2],\
                        'epislon':opt_param_adam[3], 'mini_batch_size': opt_param_adam[4]}
    super_param_rmsprop = {'lr': opt_param_rmsprop[0], 'mini_batch_size': opt_param_rmsprop[1]}
    """

    # if not perform cross validation, set the superparameters to default values
    super_param_sgd_default = {'lr': 0.01, 'mini_batch_size': 1}
    super_param_adam_default = {'lr': 1e-5, 'b1': 0.9, 'b2': 0.999,\
                        'epislon':1e-8, 'mini_batch_size': 10}
    super_param_rmsprop_default = {'lr': 0.0001, 'mini_batch_size': 10}

    relu_model.init_params()
    print('Start training for sgd:')
    train_loss_sgd = train(relu_model, train_input, train_target, Loss_MSE(), 500, True, super_param_sgd_default, 'sgd', False)
    acc = test(relu_model, test_input, test_target, True, 'sgd')
    print('Testing accuracy for sgd: {}'.format(acc))

    relu_model.init_params()
    print('Start training for adam:')
    train_loss_adam = train(relu_model, train_input, train_target, Loss_MSE(), 500, True, super_param_adam_default, 'adam', False)
    acc = test(relu_model, test_input, test_target, True, 'adam')
    print('Testing accuracy for adam: {}'.format(acc))

    relu_model.init_params()
    print('Start training for rmsprop:')
    train_loss_rmsprop = train(relu_model, train_input, train_target, Loss_MSE(), 500, True, super_param_rmsprop_default, 'rmsprop', False)
    acc = test(relu_model, test_input, test_target, True, 'rmsprop')
    print('Testing accuracy for rmsprop: {}'.format(acc))

    # plot training loss
    plot_loss(500, sgd=train_loss_sgd, adam=train_loss_adam, rmsprop = train_loss_rmsprop)


if __name__ == '__main__':
    main()