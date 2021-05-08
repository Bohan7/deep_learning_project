# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:22:21 2021

@author: Bohan
"""
import torch
import math

from model import Linear, Sequential, ReLU, Tanh, LossMSE

def generate_dict_set(size):
    if not isinstance(size, int) or size <= 0:
        raise ValueError('Size of training and testing sets must be a positive integer')
    
    data = torch.empty(size, 2).uniform_(0, 1)
    label = (data - 0.5).pow(2).sum(dim=1).sub(1 / (2 * math.pi)).sign().add(1).div(2).long()
    
    return data, label
    
def train(model, train_input, train_target, mini_batch_size, loss, nb_epochs, lr):
    criterion = loss

    for e in range(nb_epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            model.zero_grad()

            output = model.forward(train_input.narrow(0, b, mini_batch_size).t())
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size).t())
            acc_loss = acc_loss + loss

            grad_input = criterion.backward()
            model.backward(grad_input)
            model.update(lr)
        
        print('epoch {}: loss: {}'.format(e, acc_loss))

def test(model, test_input, test_target):
    prediction = model.forward(test_input.t())
    model.zero_grad()
    
    predicted_class = (prediction > 0.5).type(torch.FloatTensor)
    accuracy = (predicted_class == test_target).float().mean()

    return accuracy

def main():
    train_input, train_target = generate_dict_set(1000)
    test_input, test_target = generate_dict_set(1000)
    
    relu_model = Sequential(Linear(2, 25), ReLU(),
                            Linear(25, 25), ReLU(),
                            Linear(25, 25), ReLU(),
                            Linear(25, 1), init_type='xavier', init_gain=1.0, epsilon=1e-6)
    
    train(relu_model, train_input, train_target, 10, LossMSE(), 300, 0.001)
    acc = test(relu_model, test_input, test_target)
    print('Testing accuracy: {}'.format(acc))

if __name__ == '__main__':
    main()
    