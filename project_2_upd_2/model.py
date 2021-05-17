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
        self.vt_w = torch.empty(self.b.shape).zero_()   # for adam
        self.vt_b = torch.empty(self.b.shape).zero_()   # for adam
        self.t = 1  # for adam
        self.rt_w = torch.empty(self.w.shape).zero_()   # for rmsprop
        self.rt_b = torch.empty(self.b.shape).zero_()   # for rmsprop
        self.x_previous = None
    
    def init_params(self, init_type='zero', gain=1.0, epsilon=1e-6):
        if init_type in ['zero', 'zeros']:
            self.w.normal_(0, epsilon)
            self.b.normal_(0, epsilon)
        elif init_type == 'xavier':
            xavier_std = gain * math.sqrt(2.0 / (self.input_units + self.output_units))
            self.w.normal_(0, xavier_std)
            self.b.normal_(0, xavier_std)
        
    def forward(self, *input):
        self.x_previous = input[0].clone()
        return self.w.mm(self.x_previous) + self.b
    
    def backward(self,  *gradwrtoutput):
        grad_input = gradwrtoutput[0].clone()
        self.dl_dw.add_(grad_input.mm(self.x_previous.t()))
        self.dl_db.add_(grad_input.sum(1, True))
        
        return self.w.t().mm(grad_input)

    def param(self):
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

    def update_rmsprop(self, lr, rho=0.9, epislon=1e-10):
        for (r_t, param, grad) in [(self.rt_w, self.w, self.dl_dw), (self.rt_b, self.b, self.dl_db)]:
            r_t = rho * r_t + (1 - rho) * (grad**2)
            d_param = lr / torch.sqrt(r_t + epislon) * grad
            param.sub_(d_param)

    def zero_grad(self):
          self.dl_dw = torch.empty(self.output_units, self.input_units).zero_()
          self.dl_db = torch.empty(self.output_units, 1).zero_()

class Sequential(Module):
      def __init__(self, *modules, init_type='zero', init_gain=1.0, epsilon=1e-6):
        super(Module, self).__init__()
        self.modules = list(modules)
        self.init_type = init_type
        self.init_gain = init_gain
        self.epsilon = epsilon
        
        if init_type is not None:
            for module in self.modules:
                module.init_params(self.init_type, self.init_gain, self.epsilon)

      def init_params(self):
          for module in self.modules:
              module.init_params(self.init_type, self.init_gain, self.epsilon)

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

      def update_rmsprop(self, lr, rho=0.9, epislon=1e-10):
          for module in self.modules:
              module.update_rmsprop(lr, rho, epislon)

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
        
        return self.error.pow(2).sum()  / self.n

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

    def update_rmsprop(self, lr, rho=0.9, epislon=1e-10):
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

"""
class Batch_norm(Module):
    def __init__(self, fea_size):
        super(Batch_norm, self).__init__()
        self.s = None
        self.running_mean = 0
        self.running_var = 1
        self.momentum = 0.9
        self.epislon = 1e-8
        self.beta = torch.empty((fea_size,)).zero_()
        self.gamma = torch.ones(fea_size,)
        self.x = None
        self.d_gamma = None
        self.d_beta = None
        self.mean = None
        self.var = None
        self.std = None
        self.x_norm = None
        self.d_x = None

    def forward(self, *input):
        self.x = input[0].clone()
        self.mean = self.x.mean(axis=0)
        self.var = self.x.var(axis=0)
        self.std = self.x.std(axis=0)
        self.running_mean = (1 - self.momentum) * self.mean + self.momentum * self.running_mean
        self.running_var = (1 - self.momentum) * self.std + self.momentum * self.running_var
        self.x_norm = (self.x - self.mean) / torch.sqrt(self.var + self.epislon)
        y = self.gamma.mm(self.x_norm) + self.beta

        return y

    def backward(self, *d_y):
        grad_input = d_y[0].clone()
        self.d_beta = grad_input.sum(axis=0)
        self.d_gamma = (self.x_norm.t().mm(grad_input)).sum(axis=0)
        N = grad_input.size(0)
        self.d_x = (1 / N) * self.gamma * 1 / self.std * (
                (N * grad_input) - grad_input.sum(axis=0) - (self.x - self.mean) * (self.var + self.epislon) * (grad_input.t().mm((self.x - self.mean)).sum(axis=0)))
        return self.d_x

    def update(self, lr):
        self.gamma.sub_(lr * self.d_gamma)
        self.beta.sub_(lr * self.d_beta)

    def update_adam(self, lr, b1, b2, epislon):
        self.gamma.sub_(lr * self.d_gamma)
        self.beta.sub_(lr * self.d_beta)
"""








