# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:02:45 2021

@author: Bohan
"""
import torch
import math


class Module ( object ):
    def init_params(self, init_type, gain, epsilon):
        pass
    
    def forward (self , *input ):
        raise NotImplementedError
        
    def backward (self , *gradwrtoutput ):
        raise NotImplementedError
        
    def param (self):
        return []
    
    def update (self, lr):
        pass
    
    def zero_grad (self):
        pass

    
    
    
class Linear (Module):
    def __init__(self, input_units, output_units):
        super(Linear, self).__init__()
        self.input_units = input_units
        self.output_units = output_units
        self.w = torch.empty(self.output_units, self.input_units)
        self.b = torch.empty(self.output_units, 1)
        self.dl_dw = torch.empty(self.output_units, self.input_units).zero_()
        self.dl_db = torch.empty(self.output_units, 1).zero_()
        self.x_previous = None
    
    def init_params(self, init_type='zero', gain=1.0, epsilon=1e-6):
        if init_type in ['zero', 'zeros']:
            self.w.normal_(0, epsilon)
            self.b.normal_(0, epsilon)
        elif init_type == 'xavier':
            xavier_std = gain * math.sqrt(2.0 / (self.input_units + self.output_units))
            self.w.normal_(0, xavier_std)
            self.b.normal_(0, xavier_std)
        
    def forward (self , *input):
        self.x_previous = input[0].clone()
        return self.w.mm(self.x_previous) + self.b
    
    def backward (self ,  *gradwrtoutput):
        grad_input = gradwrtoutput[0].clone()
        self.dl_dw.add_(grad_input.mm(self.x_previous.t()))
        self.dl_db.add_(grad_input.sum(1, True))
        
        return self.w.t().mm(grad_input)
        
    def param (self):
        return [(self.w, self.dl_dw), (self.b, self.dl_db)]
    
    def update(self, lr):
        self.w.sub_(lr * self.dl_dw)
        self.b.sub_(lr * self.dl_db)
    
    def zero_grad(self):
          self.dl_dw = torch.empty(self.output_units, self.input_units).zero_()
          self.dl_db = torch.empty(self.output_units, 1).zero_()

class Sequential(Module):
      def __init__(self, *modules, init_type='zero', init_gain=1.0, epsilon=1e-6):
        super(Module, self).__init__()
        self.modules = list(modules)
        
        if init_type is not None:
            for module in self.modules:
                module.init_params(init_type, init_gain, epsilon)
                
      def forward (self , *input):
        x = input[0].clone()
        
        for module in self.modules:
                x = module.forward(x)
        return x

      def backward (self , *gradwrtoutput):
        grad_input = gradwrtoutput[0].clone()
        
        for module in self.modules[::-1]:
                grad_input = module.backward(grad_input)
                
      def param (self):
          params = []
          
          for module in self.modules:
            for param in module.param():
                params.append(param)
                
          return params
      
      def update(self, lr):
          for module in self.modules:
              module.update(lr)
      
      def zero_grad(self):
          for module in self.modules:
              module.zero_grad()

class LossMSE(Module):
    def __init__(self):
        super(Module, self).__init__()
        self.error = None
        self.n = None
        
    def forward(self, predict, target):
        f_x = predict.clone()
        y = target.clone()
        self.error = f_x - y
        self.n = y.size(0)
        
        return self.error.pow(2).sum() / self.n
    
    def backward(self):
        return 2 * self.error / self.n

class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.s = None
        
    def forward (self , *input):
         self.s = input[0].clone()
         return(self.s.relu())
     
    def backward (self ,  *gradwrtoutput):
         grad_input = gradwrtoutput[0].clone()
         
         s_track_grad = self.s.clone()
         s_track_grad[s_track_grad > 0] = 1.0
         s_track_grad[s_track_grad < 0] = 0.0
         
         return s_track_grad * grad_input

class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.s = None
        
    def forward (self , *input):
         self.s = input[0].clone()
         return(self.s.tanh())
     
    def backward (self ,  *gradwrtoutput):
         grad_input = gradwrtoutput[0].clone()
         
         s_track_grad = self.s.clone()
         s_track_grad = 4 * (s_track_grad.exp() + s_track_grad.mul(-1).exp()).pow(-2)
         
         return s_track_grad * grad_input 

