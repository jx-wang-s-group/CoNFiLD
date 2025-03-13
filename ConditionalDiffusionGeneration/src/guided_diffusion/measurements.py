'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
import torch
from ConditionalNeuralField.cnf.inference_function import pass_through_model_batch
from ConditionalNeuralField.cnf.utils.normalize import Normalizer_ts
from ConditionalNeuralField.cnf.nf_networks import SIRENAutodecoder_film
import numpy as np
from einops import rearrange
# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper

class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)

class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data.to(self.device) * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)
    
@register_operator(name='case2')
class Case2Operator(NonLinearOperator):
    def __init__(self, device,
                 ckpt_path,
                 max_val,
                 min_val,
                 coords,
                 batch_size):
        
        self.device = device
        self.coords = torch.tensor(coords, dtype = torch.float32, device=device)

        self.x_normalizer = Normalizer_ts(method = '-11',dim=0,
                                    params = [torch.tensor([1.,1.], device = device),
                                            torch.tensor([0.,0.], device = device)])
        self.y_normalizer = Normalizer_ts(method = '-11',dim=0, 
                                    params = [torch.tensor([[0.9617, 0.2666, 0.2869, 0.0290]], device = device), 
                                            torch.tensor([[-0.0051, -0.2073, -0.2619, -0.0419]], device = device)])
        cin_size, cout_size = 2,4
        self.model = SIRENAutodecoder_film(cin_size,256,cout_size,10,256)
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        self.max_val = torch.from_numpy(max_val).to(device) 
        self.min_val = torch.from_numpy(min_val).to(device)
        
        self.batch_size = batch_size

    def _unnorm(self, norm_data):
        return ((norm_data[:, 0, ...] + 1)*(self.max_val- self.min_val)/2 + self.min_val)[:, None, ...]

    def forward(self, data, **kwargs):
        mask = kwargs.get('mask', None)
        data_reshaped = rearrange(self._unnorm(data), "s c t l -> (s c t) l")
        phy_fields = pass_through_model_batch(self.coords, data_reshaped, self.model, 
                                              self.x_normalizer, self.y_normalizer,
                                              self.batch_size, self.device)
        return mask*phy_fields

@register_operator(name='case3')
class Case3Operator(NonLinearOperator):
    def __init__(self, device,
                 coords,
                 batch_size,
                 max_val,
                 min_val,
                 normalizer_params_path,
                 ckpt_path) -> None:
        
        self.device = device
        self.coords = torch.tensor(coords, dtype = torch.float32, device=device)
        
        params = torch.load(normalizer_params_path)
        x_ub,x_lb = params['x_normalizer_params']
        y_ub,y_lb = params['y_normalizer_params']
        cin_size, cout_size = 2,2
        self.x_normalizer = Normalizer_ts(method = '-11',dim=0, params = (x_ub,x_lb))
        self.y_normalizer = Normalizer_ts(method = '-11',dim=0, params = (y_ub[:cout_size],y_lb[:cout_size]))
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_film(cin_size,256,cout_size,17,256)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        self.max_val = torch.from_numpy(max_val).to(device) 
        self.min_val = torch.from_numpy(min_val).to(device)
        
        self.batch_size = batch_size
        
    def _unnorm(self, norm_data):
        print("DEBUG: _unnorm -> norm_data.shape =", norm_data.shape)
        print("DEBUG: self.max_val.shape =", self.max_val.shape, " self.min_val.shape =", self.min_val.shape)
        return ((norm_data[:, 0, ...] + 1)*(self.max_val- self.min_val)/2 + self.min_val)[:, None, ...] 
    
    def forward(self, data, **kwargs):
        data_reshaped = rearrange(self._unnorm(data), "s c t l -> (s c t) l")
        return pass_through_model_batch(self.coords, data_reshaped, self.model, 
                                        self.x_normalizer, self.y_normalizer,
                                        self.batch_size, self.device)
        
@register_operator(name='case3_gappy')
class Case3Operator_gappy(NonLinearOperator):
    def __init__(self, device,
                 coords,
                 batch_size,
                 max_val,
                 min_val,
                 normalizer_params_path,
                 ckpt_path
                 ) -> None:
        
        self.device = device
        self.coords = torch.tensor(coords, dtype = torch.float32, device=device)
        
        params = torch.load(normalizer_params_path)
        x_ub,x_lb = params['x_normalizer_params']
        y_ub,y_lb = params['y_normalizer_params']
        cin_size, cout_size = 2,2
        self.x_normalizer = Normalizer_ts(method = '-11',dim=0, params = (x_ub,x_lb))
        self.y_normalizer = Normalizer_ts(method = '-11',dim=0, params = (y_ub[:cout_size],y_lb[:cout_size]))
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_film(cin_size,256,cout_size,17,256)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        self.max_val = torch.from_numpy(max_val).to(device) 
        self.min_val = torch.from_numpy(min_val).to(device)
        
        self.batch_size = batch_size
        
    def _unnorm(self, norm_data):
        return ((norm_data[:, 0, ...] + 1)*(self.max_val- self.min_val)/2 + self.min_val)[:, None, ...] 
    
    def forward(self, data, **kwargs):
        data_reshaped = rearrange(self._unnorm(data), "s c t l -> (s c t) l")
        out =  pass_through_model_batch(self.coords, data_reshaped, self.model, 
                                              self.batch_size, self.x_normalizer, self.y_normalizer,
                                              self.device)
        out[:, :10, 1] = 0.
        out[:,10:, 0] = 0.
        return out

@register_operator(name='case4')
class Case4Operator(NonLinearOperator):
    def __init__(self, device,
                 coords_path,
                 batch_size,
                 max_val_path,
                 min_val_path,
                 normalizer_params_path,
                 ckpt_path
                 ) -> None:
        
        self.device = device
        coords = np.load(coords_path)
        self.coords = torch.tensor(coords, dtype = torch.float32, device=device)
        
        params = torch.load(normalizer_params_path)
        x_uub, x_llb = params['x_normalizer_params']
        y_uub,_ = params['y_normalizer0u_params']
        _,y_llb = params['y_normalizer0l_params']
        cin_size, cout_size = 3,3
        self.x_normalizer = Normalizer_ts(method = '-11',dim=0, params = (x_uub,x_llb))  # only take out xyz 
        self.y_normalizer = Normalizer_ts(method = '-11',dim=0, params = (y_uub[:cout_size],y_llb[:cout_size]))
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_film(cin_size,384,cout_size,15,384) 
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        max_val = np.load(max_val_path)
        min_val = np.load(min_val_path)
        self.max_val = torch.from_numpy(max_val).to(device) 
        self.min_val = torch.from_numpy(min_val).to(device)
        
        self.batch_size = batch_size
        
    def _unnorm(self, norm_data):
        return ((norm_data[:, 0, ...] + 1)*(self.max_val- self.min_val)/2 + self.min_val)[:, None, ...] 
    
    def forward(self, data, **kwargs):
        data_reshaped = rearrange(self._unnorm(data), "s c t l -> (s c t) l")
        return pass_through_model_batch(self.coords, data_reshaped, self.model, 
                                            self.x_normalizer, self.y_normalizer, self.batch_size,
                                              self.device)
# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma

@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)