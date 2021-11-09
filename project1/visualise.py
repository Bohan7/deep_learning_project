import random
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from model import CNN, Siamese_net, Resnetblock, Resnetblock_WS
import matplotlib.pyplot as plt
import numpy as np

def plot_train_loss_test_accuracy(loss, accuracy, model_names):
  train_loss = np.array(loss)
  test_accuracy = np.array(accuracy)

  epochs = 25
  values = [train_loss, test_accuracy]
  values_labels = ['Train Loss', 'Test accuracy']

  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
  plt.suptitle('Comparison of different models')

  for i, (y_label, value) in enumerate(zip(values_labels, values)):
    for model, plot_value in zip(model_names, value):
      values_mean = np.mean(plot_value, axis=0)
      values_std = np.std(plot_value, axis=0)
      
      if y_label == 'Test accuracy':
        print('model {}, Test accuracy: {:.2f} +/- {:.2f}'.format(model, values_mean[-1], values_std[-1]))

      axes[i].plot(range(epochs), values_mean, label=model)
      axes[i].fill_between(range(epochs), values_mean - values_std, values_mean + values_std, alpha=0.3)
      axes[i].set_xlabel('Epoch')
      axes[i].set_ylabel(y_label)
      axes[i].legend()
  
  plt.savefig('Comparison of different models' + ".png")
  plt.show()
    
