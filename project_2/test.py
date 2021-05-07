# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:10:41 2021

@author: Bohan
"""
import torch

def generate_disc_set(nb):
    input = torch.empty(nb, 2).uniform_(-1, 1)
    target = input.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2).long()
    return input, target