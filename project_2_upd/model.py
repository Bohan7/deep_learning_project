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
        self.mt_w = torch.empty(self.w.shape).zero_()   # for adam
        self.mt_b = torch.empty(self.b.shape).zero_()   # for adam
        self.vt_w = torch.empty(self.b.shape).zero_()
        self.vt_b = torch.empty(self.b.shape).zero_()
        self.t = 1
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

    def update_adam(self, lr, b1, b2, epislon):
        for (mt,vt,param,grad) in [(self.mt_w, self.vt_w, self.w, self.dl_dw), (self.mt_b, self.vt_b, self.b, self.dl_db)]:
            mt = b1 * mt + (1 - b1) * grad
            vt = b2 * vt + (1 - b2) * (grad ** 2)
            mt_hat = mt / (1 - (b1 ** self.t))
            vt_hat = vt / (1 - (b2 ** self.t))
            theta = lr * mt_hat / (vt_hat.sqrt() + epislon)
            param.sub_(theta)
        self.t += 1


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

      def update_adam(self, lr, b1, b2, epislon):
          for module in self.modules:
              module.update_adam(lr, b1, b2, epislon)
      
      def zero_grad(self):
          for module in self.modules:
              module.zero_grad()

class Loss_MSE(Module):
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

class Loss_Cross_Entropy(Module):
    def __int__(self):
        super(Module, self).__init__()
        self.n = None
        self.f_x = None
        self.y = None
    def forward(self, predict, target):
        self.f_x = predict.clone()
        self.y = target.clone()
        self.n = self.y.size(0)

        loss_sum = sum(list(map(lambda p,q: p*torch.log(q) + (1-p)*torch.log(1-q), self.y, self.f_x)))
        return - loss_sum / self.n
    def backward(self):
        loss_grad = self.y.div(self.f_x) - (1-self.y).div(self.f_x)
        return - loss_grad / self.n

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

    def update_adam(self, lr, b1, b2, epislon):
        pass

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

