# -*- coding: utf-8 -*-
"""
Created on Tue May 25 13:40:54 2021

@author: Bohan
"""
import random
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from model import CNN, Siamese_net, Resnetblock, Resnetblock_WS
from visualise import plot_train_loss_test_accuracy

import matplotlib.pyplot as plt
from dlc_practical_prologue import generate_pair_sets
from test import standardize, train_CNN, train_Siamese_net, test_CNN, test_Siamese_net

def cross_validate(model_name, model, train_input, train_target, train_classes, sup_param_range):
  cuda = torch.cuda.is_available()
  device = torch.device('cuda' if cuda else 'cpu')
  # retrieve the range of the superparameters from the dictionary
  lr_range = sup_param_range['lr_range']
  mini_batch_size_range = sup_param_range['mini_batch_size_range']
  use_bn_range = sup_param_range['use_bn']

  N = 1000
  cross_entropy = nn.CrossEntropyLoss() 

  if 'auxiliary' in model_name:
    weight_range = sup_param_range['weight_range']
    params_range = [(lr, mini_batch_size, use_bn, weight) for lr in lr_range for mini_batch_size in mini_batch_size_range for use_bn in use_bn_range for weight in weight_range]
  else:
    params_range = [(lr, mini_batch_size, use_bn) for lr in lr_range for mini_batch_size in mini_batch_size_range for use_bn in use_bn_range]

  # use 5-fold cross validation
  n_fold = 5
  fold_size = int(len(train_input)/n_fold)
  val_acc_mean = []
  print("Cross validation starts for " + str(model_name))
  # iterate over the parameters to find the optimal superparameters
  for params in params_range:
    val_acc = []
    if 'CNN' in model_name:
      net = model(in_channels=2, out_channels_1=32, out_channels_2=64, output_fc=25, use_bn=params[2]).to(device)
    elif 'version1' in model_name:
      net = model(in_channels=2, out_channels_1=16, out_channels_2=32, output_fc1=50, output_fc2=25, use_bn=params[2], version=1).to(device)
    elif 'version2' in model_name:
      net = model(in_channels=2, out_channels_1=16, out_channels_2=32, output_fc1=50, output_fc2=25, use_bn=params[2], version=2).to(device)
    elif 'Resnet' in model_name:
      net = model(in_channels=2, out_channels_1=16, out_channels_2=16, output_fc=80, kernel_size=3, use_bn=params[2]).to(device)

    optimizer = optim.SGD(net.parameters(), lr=params[0])

    for i in range(n_fold):
      # split to training and validation set
      train_set = torch.cat((train_input[0:i*fold_size, :], train_input[(i+1)*fold_size: len(train_input):, :]), 0)
      train_set_target = torch.cat((train_target[0: i*fold_size], train_target[(i+1)*fold_size: len(train_target)]), 0)
      train_set_classes = torch.cat((train_classes[0: i*fold_size], train_classes[(i+1)*fold_size: len(train_target)]), 0)

      val_set = train_input[i*fold_size : (i+1)*fold_size, :]
      val_set_target = train_target[i*fold_size : (i+1)*fold_size]
      val_set_classes = train_classes[i*fold_size : (i+1)*fold_size]

      if ('weight_share' in model_name) or (model_name is 'Siamese_version1'):
        train_Siamese_net(net, train_set, train_set_target, train_set_classes, None, None, None, 100, cross_entropy, optimizer, 25, [1,0], version=1, verbose=False)
        val_acc.append(test_Siamese_net(net, val_set, val_set_target, version=1))
      elif 'auxiliary' in model_name:
        train_Siamese_net(net, train_set, train_set_target, train_set_classes, None, None, None, 100, cross_entropy, optimizer, 25, params[3], version=1, verbose=False)
        val_acc.append(test_Siamese_net(net, val_set, val_set_target, version=1))
      elif model_name is 'Siamese_version2':
        train_Siamese_net(net, train_set, train_set_target, train_set_classes, None, None, None, 100, cross_entropy, optimizer, 25, [0,1], version=2, verbose=False)
        val_acc.append(test_Siamese_net(net, val_set, val_set_target, version=2))
      else:
        train_CNN(net, train_set, train_set_target, None, None, 100, cross_entropy, optimizer, 25, verbose=False)
        val_acc.append(test_CNN(net, val_set, val_set_target))

    
    val_acc_mean.append(sum(val_acc)/n_fold)

    # print validation result for each combination of superparameters
    if 'auxiliary' in model_name:
      print("lr: {}, mini_batch_size : {}, use_bn:{}, weight : {}, validation accuracy is {}".format(params[0], params[1], params[2], params[3], float(sum(val_acc) / n_fold)))
    else:
      print("lr: {}, mini_batch_size : {}, use_bn: {}, validation accuracy is {}".format(params[0], params[1], params[2], float(sum(val_acc) / n_fold)))
        
    

  # find the optimal parameters (with highest average validation accuracy)
  argmax_acc = [i for i, val in enumerate(val_acc_mean) if (val == max(val_acc_mean))][0]
  opt_param = params_range[argmax_acc]
  opt_val_acc = val_acc_mean[argmax_acc]

  # print the optimal parameters found by cross validation
  if 'auxiliary' in model_name:
      print("Best parameter: for {}  lr: {}, mini_batch_size : {}, use_bn:{}, weight : {}, validation accuracy is {}".format(model_name,opt_param[0], opt_param[1], opt_param[2], opt_param[3], opt_val_acc.item()))
  else:
      print("Best parameter: for {}  lr: {}, mini_batch_size : {}, use_bn: {}, validation accuracy is {}".format(model_name, opt_param[0], opt_param[1], opt_param[2], opt_val_acc.item()))

  return opt_param




def main():
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    N = 1000
    n_run = 15
    test_accuracy = torch.zeros(n_run)
    cross_entropy = nn.CrossEntropyLoss()


    train_input, train_class, train_digit, test_input, test_class, test_digit = generate_pair_sets(N)

    if torch.cuda.is_available():
        train_input, train_class, train_digit = train_input.to(device), train_class.to(device), train_digit.to(device)
        test_input, test_class, test_digit = test_input.to(device), test_class.to(device), test_digit.to(device)
    
    train_input, test_input = standardize(train_input, test_input)

    model_names = ['CNN', 'Siamese_version1', 'Siamese_version1_auxiliary_loss', 'Siamese_version2', 'Resnet_block', 'Resnet_block_weight_share', 'Resnet_block_weight_share_auxiliary_loss']
    sup_param_cv_range = {'lr_range' : [1e-1, 1e-2, 1e-3, 1e-4],
                   'mini_batch_size_range':[50, 100, 200],
                   'use_bn':[True, False],
                   'weight_range': [[1, 1 + x] for x in (0, -0.25, -0.5, -0.75, 0.25, 0.5, 1)]
                }
    
    for name in model_names:
      if 'CNN' in name:
       model = CNN
      elif 'Siamese' in name:
       model = Siamese_net
      elif 'weight_share' in name:
       model = Resnetblock_WS
      elif 'Resnet' in name:
       model = Resnetblock
      
      opt_param = cross_validate(name, model, train_input, train_class, train_digit, sup_param_cv_range)

if __name__ == '__main__':
    main()
    