# -*- coding: utf-8 -*-
"""
Created on Sat May 22 23:58:50 2021

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

def standardize(train_data, test_data):
    mean, std = train_data.mean(), train_data.std()
    return (train_data - mean) / std, (test_data - mean) / std

def train_CNN(model, train_input, train_target, test_input, test_target, mini_batch_size, criterion, optimizer, nb_epochs, verbose=True):
  loss_values = []
  acc_values = []

  for e in range(nb_epochs):
    train_loss = 0
    for b in range(0, train_input.size(0), mini_batch_size):
      model.zero_grad()
      output = model(train_input.narrow(0, b, mini_batch_size))
      loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
      loss.backward()
      optimizer.step()

      train_loss += loss.item()

    
    
    if verbose:
      acc = test_CNN(model, test_input, test_target)
      print('epoch {}:   loss: {}    acc: {}'.format(e, train_loss, acc))
      loss_values.append(train_loss)
      acc_values.append(acc.item())
    
    if (e > 5) and ((loss_values[-1] - loss_values[-2]) > 10.0):
      loss_values = loss_values[:-1] + [loss_values[-2]] * (nb_epochs-len(loss_values[:-1]))
      acc_values = acc_values[:-1] + [acc[-2]] * (nb_epochs-len(acc_values[:-1]))
      break
    

  return loss_values, acc_values

def train_Siamese_net(model, train_input, train_target, train_classes, test_input, test_target, test_classes, mini_batch_size, criterion, optimizer, nb_epochs, loss_weight, version=1, verbose=True):
  loss_values = []
  acc_values = []

  for e in range(nb_epochs):
    train_loss = 0
    for b in range(0, train_input.size(0), mini_batch_size):
      model.zero_grad()
      output, (pred_class_1, pred_class_2) = model(train_input.narrow(0, b, mini_batch_size))

      loss_class_1 = criterion(pred_class_1, train_classes.narrow(0, b, mini_batch_size)[:,0])
      loss_class_2 = criterion(pred_class_2, train_classes.narrow(0, b, mini_batch_size)[:,1])
      
      if version == 1:
        loss_bool = criterion(output, train_target.narrow(0, b, mini_batch_size))
        total_loss = loss_weight[0] * loss_bool + loss_weight[1] * (loss_class_1+loss_class_2)
      elif version == 2:
        total_loss = loss_weight[1] * (loss_class_1+loss_class_2)

      total_loss.backward()
      optimizer.step()

      train_loss += total_loss.item()

      
    
    if verbose and (version==1):
      acc = test_Siamese_net(model, test_input, test_target, version=1)
      print('epoch: {}   loss: {}   acc: {}'.format(e, train_loss, acc))
    elif verbose and (version==2):
      acc = test_Siamese_net(model, test_input, test_target, version=2)
      print('epoch: {}   loss: {}   acc: {}'.format(e, train_loss, acc))
    
    loss_values.append(train_loss)
    acc_values.append(acc.item())

    if (e > 5) and ((loss_values[-1] - loss_values[-2]) > 10.0):
      loss_values = loss_values[:-1] + [loss_values[-2]] * (nb_epochs-len(loss_values[:-1]))
      acc_values = acc_values[:-1] + [acc_values[-2]] * (nb_epochs-len(acc_values[:-1]))
      break
    
  return loss_values, acc_values



def test_CNN(model, test_input, test_target):  
    model.eval()
    with torch.no_grad():
        pred = model(test_input)
    
    _, predicted = pred.max(1)
    accuracy = (predicted == test_target).float().mean()

    return accuracy

def test_Siamese_net(model, test_input, test_target, version=1):  
    model.eval()
    with torch.no_grad():
        pred, (pred_class_1, pred_class_2) = model(test_input)
    
    if version == 1:
      _, predicted = pred.max(1)
      accuracy = (predicted == test_target).float().mean()
    elif version == 2:
      accuracy = (pred == test_target).float().mean()

    return accuracy

def main():
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    N = 1000
    n_run = 15
    test_accuracy = torch.zeros(n_run)
    cross_entropy = nn.CrossEntropyLoss()

    model_names = ['CNN', 'Siamese_version1', 'Siamese_version1_auxiliary_loss', 'Siamese_version2', 'Resnet_block', 'Resnet_block_weight_share', 'Resnet_block_weight_share_aux']

    models_loss = []
    models_acc = []
    for name in model_names:
      runs_loss = []
      runs_accuracy = []
      for i in range(n_run):
        seed = random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


        train_input, train_class, train_digit, test_input, test_class, test_digit = generate_pair_sets(N)

        if torch.cuda.is_available():
          train_input, train_class, train_digit = train_input.to(device), train_class.to(device), train_digit.to(device)
          test_input, test_class, test_digit = test_input.to(device), test_class.to(device), test_digit.to(device)
    
        train_input, test_input = standardize(train_input, test_input)

        print('\nrun {}!'.format(i))
        
        if 'CNN' is name:
          print('CNN')
          model = CNN(in_channels=2, out_channels_1=32, out_channels_2=64, output_fc=25, use_bn=True).to(device)
          optimizer = optim.SGD(model.parameters(), lr = 1e-1)
          train_loss, test_acc = train_CNN(model, train_input, train_class, test_input, test_class, 100, cross_entropy, optimizer, 25)
        elif 'Siamese_version1' is name:
          print(name)
          model = Siamese_net(in_channels=2, out_channels_1=16, out_channels_2=32, output_fc1=50, output_fc2=25, use_bn=True, version=1).to(device)
          optimizer = optim.SGD(model.parameters(), lr = 1e-1)
          train_loss, test_acc = train_Siamese_net(model, train_input, train_class, train_digit, test_input, test_class, test_digit, 100, cross_entropy, optimizer, 25, [1,0], version=1)
        elif 'Siamese_version1_auxiliary_loss' is name:
          print(name)
          model = Siamese_net(in_channels=2, out_channels_1=16, out_channels_2=32, output_fc1=50, output_fc2=25, use_bn=True, version=1).to(device)
          optimizer = optim.SGD(model.parameters(), lr = 1e-1)
          train_loss, test_acc = train_Siamese_net(model, train_input, train_class, train_digit, test_input, test_class, test_digit, 100, cross_entropy, optimizer, 25, [1,1], version=1)
         
        elif 'Siamese_version2' is name:
          print(name)
          model = Siamese_net(in_channels=2, out_channels_1=16, out_channels_2=32, output_fc1=50, output_fc2=25, use_bn=True, version=2).to(device)
          optimizer = optim.SGD(model.parameters(), lr = 1e-1)
          train_loss, test_acc = train_Siamese_net(model, train_input, train_class, train_digit, test_input, test_class, test_digit, 100, cross_entropy, optimizer, 25, [0, 1], version=2)
          
        elif 'Resnet_block' is name:
          print(name)
          model = Resnetblock(in_channels=2, out_channels_1=16, out_channels_2=16, output_fc=80, kernel_size=3, use_bn=True).to(device)
          optimizer = optim.SGD(model.parameters(), lr = 1e-1)
          train_loss, test_acc = train_CNN(model, train_input, train_class, test_input, test_class, 100, cross_entropy, optimizer, 25)
        elif 'Resnet_block_weight_share' is name:
          print(name)
          model = Resnetblock_WS(in_channels=2, out_channels_1=16, out_channels_2=16, output_fc=80, kernel_size=3, use_bn=True).to(device)
          optimizer = optim.SGD(model.parameters(), lr = 1e-1)
          train_loss, test_acc = train_Siamese_net(model, train_input, train_class, train_digit, test_input, test_class, test_digit, 100, cross_entropy, optimizer, 25, [1,0], version=1)
         
        elif 'Resnet_block_weight_share_aux' is name:
          print(name)
          model =  Resnetblock_WS(in_channels=2, out_channels_1=16, out_channels_2=16, output_fc=80, kernel_size=3, use_bn=True).to(device)
          optimizer = optim.SGD(model.parameters(), lr = 1e-1)
          train_loss, test_acc = train_Siamese_net(model, train_input, train_class, train_digit, test_input, test_class, test_digit, 100, cross_entropy, optimizer, 25, [1,1], version=1)

        if i==0:
          total_params = sum(p.numel() for p in model.parameters())
          print('There are {} params'.format(total_params))
        
        
        runs_loss.append(train_loss)
        runs_accuracy.append(test_acc)

      models_loss.append(runs_loss)
      models_acc.append(runs_accuracy)
      

      #mean = test_accuracy.mean().item()
      #std = test_accuracy.std().item()
      #print(f'Test accuracy: {mean:.2f} +- {std:.2f}')
    plot_train_loss_test_accuracy(models_loss, models_acc, model_names)


    
if __name__ == '__main__':
    main()
    