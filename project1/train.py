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
from model import CNN, CNN_WS, Resnetblock

import matplotlib.pyplot as plt
from dlc_practical_prologue import generate_pair_sets

def standardize(train_data, test_data):
    mean, std = train_data.mean(), train_data.std()
    return (train_data - mean) / std, (test_data - mean) / std

def train_CNN(model, train_input, train_target, mini_batch_size, criterion, optimizer, nb_epochs):
  for e in range(nb_epochs):
    train_loss = 0
    for b in range(0, train_input.size(0), mini_batch_size):
      model.zero_grad()
      output = model(train_input.narrow(0, b, mini_batch_size))
      loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
    
    print('epoch {}: loss: {}'.format(e, train_loss))

def train_CNN_WS(model, train_input, train_target, mini_batch_size, criterion, optimizer, nb_epochs):
  for e in range(nb_epochs):
    train_loss = 0
    for b in range(0, train_input.size(0), mini_batch_size):
      model.zero_grad()
      output, (pred_class_1, pred_class_2) = model(train_input.narrow(0, b, mini_batch_size))
      loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
    
    print('epoch {}: loss: {}'.format(e, train_loss))

def train_CNN_WS_AL(model, train_input, train_target, train_classes, mini_batch_size, criterion, optimizer, nb_epochs, loss_weight):
  for e in range(nb_epochs):
    train_loss = 0
    for b in range(0, train_input.size(0), mini_batch_size):
      model.zero_grad()
      output, (pred_class_1, pred_class_2) = model(train_input.narrow(0, b, mini_batch_size))
      loss_bool = criterion(output, train_target.narrow(0, b, mini_batch_size))
      loss_class_1 = criterion(pred_class_1, train_classes.narrow(0, b, mini_batch_size)[:,0])
      loss_class_2 = criterion(pred_class_2, train_classes.narrow(0, b, mini_batch_size)[:,1])
      
      total_loss = loss_weight[0] * loss_bool + loss_weight[1] * (loss_class_1+loss_class_2)
      total_loss.backward()
      optimizer.step()

      train_loss += total_loss.item()
    
    print('epoch {}: loss: {}'.format(e, train_loss))


def test_CNN(model, test_input, test_target):  
    model.eval()
    with torch.no_grad():
        pred = model(test_input)
    
    _, predicted = pred.max(1)
    accuracy = (predicted == test_target).float().mean()

    return accuracy

def test_CNN_WS(model, test_input, test_target):  
    model.eval()
    with torch.no_grad():
        pred, (pred_class_1, pred_class_2) = model(test_input)
    
    _, predicted = pred.max(1)
    accuracy = (predicted == test_target).float().mean()

    return accuracy

def main():
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    N = 1000
    n_run = 20
    test_accuracy = torch.zeros(n_run)
    cross_entropy = nn.CrossEntropyLoss()
    
    for i in range(n_run):
      seed = random.randint(1, 10000)
      random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      

      model = CNN_WS(in_channels=2, out_channels_1=32, out_channels_2=64, output_fc=256, use_bn=True).to(device)
      #model = Resnetblock(in_channels=2, out_channels_1=64, out_channels_2=64, output_fc=50, kernel_size=3, use_bn=True)
      optimizer = optim.SGD(model.parameters(), lr = 1e-1)

      if i==0:
        total_params = sum(p.numel() for p in model.parameters())
        print('There are {} params'.format(total_params))

      train_input, train_class, train_digit, test_input, test_class, test_digit = generate_pair_sets(N)

      if torch.cuda.is_available():
        train_input, train_class, train_digit = train_input.to(device), train_class.to(device), train_digit.to(device)
        test_input, test_class, test_digit = test_input.to(device), test_class.to(device), test_digit.to(device)
    
      train_input, test_input = standardize(train_input, test_input)

      print('\nrun {}!'.format(i))
      train_CNN_WS_AL(model, train_input, train_class, train_digit, 100, cross_entropy, optimizer, 25, [1, 1])
      acc = test_CNN_WS(model, test_input, test_class)
      test_accuracy[i] = acc
      print(acc)
    mean = test_accuracy.mean().item()
    std = test_accuracy.std().item()
    print(f'Test accuracy: {mean:.2f} +- {std:.2f}')

if __name__ == '__main__':
    main()
    