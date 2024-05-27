import torch
import numpy as np
import math
import torch.autograd.functional as A
from einops import rearrange
from random import random
import time

def local_basis(model, seed=None, pooling_kernel=2, sample=None, t=0, shape=(1,3,256,256), revert_to_ori=True):
    '''
    This code is from https://github.com/isno0907/localbasis
    '''
    pooling_layer = torch.nn.AvgPool2d(pooling_kernel)
    gen = torch.Generator(device=model.device)
    if sample is None and seed is not None:
        sample = torch.randn(shape, generator=gen.manual_seed(seed), device=model.device)
    else:
        sample = torch.randn(shape, device=model.device)
    if hasattr(model, "forward_H"):
        J = A.jacobian(lambda x: pooling_layer(model.forward_H(x, t)), sample)
        h_b, h_c, h_h, h_w, x_b, x_c, x_h, x_w = J.shape
    else:
        raise ("Unet does not have forward_H.")

    J = torch.diagonal(J, 0, dim1=0, dim2=4)
    h_basis, s, x_basis= torch.svd(rearrange(J, "hc hh hw xc xh xw b -> b (hc hh hw) (xc xh xw)"))
    if revert_to_ori:
        x_basis = rearrange(x_basis, "b (c h w) d -> b c h w d", c=x_c, h=x_h, w=x_w) # Revert to its original form. [(c h w), d] -> [c, h, w, d]
        h_basis = rearrange(h_basis, "b (c h w) d -> b c h w d", c=h_c, h=h_h, w=h_w)
    del J
    torch.cuda.empty_cache()
        
    return sample, h_basis, s, x_basis

def compute_grsm_metric(local_basis_1, local_basis_2, d = 1, metric_type = 'geodesic'):
    assert(metric_type in ['proj', 'geodesic'])
    if metric_type == 'geodesic':
        metric = _metric_by_geodesic(local_basis_1, local_basis_2, subspace_dim = d)
    else:
        metric = _metric_by_proj_matrix(local_basis_1, local_basis_2, subspace_dim = d)
    return metric

def _metric_by_proj_matrix(local_basis_1, local_basis_2, subspace_dim):
    local_basis_1 = rearrange(local_basis_1, "c h w d -> (c h w) d")
    local_basis_2 = rearrange(local_basis_2, "c h w d -> (c h w) d")
    proj_1 = np.array(local_basis_1[:, :subspace_dim])
    proj_1 = np.matmul(proj_1, proj_1.transpose())
    proj_2 = np.array(local_basis_2[:, :subspace_dim])
    proj_2 = np.matmul(proj_2, proj_2.transpose())
    
    metric = np.linalg.norm(proj_1 - proj_2, ord = 2)
    return metric

def _metric_by_geodesic(local_basis_1, local_basis_2, subspace_dim):
    local_basis_1 = rearrange(local_basis_1, "c h w d -> (c h w) d")
    local_basis_2 = rearrange(local_basis_2, "c h w d -> (c h w) d")
    subspace_1 = np.array(local_basis_1[:, :subspace_dim])
    subspace_2 = np.array(local_basis_2[:, :subspace_dim])
    
    u, s, v = np.linalg.svd(np.matmul(subspace_1.transpose(), subspace_2))
    s[s > 1] = 1
    s = np.arccos(s)
    return np.linalg.norm(s)


def stereographic_proj(x, device):
    # R x S^(n-1) -> R x R^(n-1)
    bs = x.shape[0]
    x = x.flatten(1)
    r = (x**2).sum(1, keepdim=True).sqrt()

    return torch.cat((r, (1/(r - x[:,-1:])) * x[:,:-1]), dim=-1)

def inv_stereographic_proj(t, shape ,device):
    #  R x R^(n-1) --> R x S^(n-1)
    t = t.flatten(1)
    r, t = t[:,:1], t[:,1:]
    t_norm_sq = (t ** 2).sum(1, keepdim=True)

    result = (r / (t_norm_sq + 1)) * torch.cat((2 * t, t_norm_sq - 1), dim=-1) 

    return result.reshape(shape)

def riemmanian_metric(t, device):
    # Riemmanian metric in stereographic coordinates
    r, t = t[:,:1], t[:,1:]
    t_norm_sq = (t ** 2).sum(1, keepdim=True)  

    G_r = torch.ones((t.shape[0], 1), device=device) # B, 1
    G_t = 4 * r ** 4 / ((t_norm_sq + r ** 2) ** 2) * torch.ones_like(t) # B, x-1

    return torch.cat((G_r, G_t), dim=-1) # B, x

def inv_stereographic_proj_t(t):
    '''
    t: torch.tensor whose size = (n, D-1)
    '''
    t = t.flatten(1)
    t_norm_sq = (t ** 2).sum(1, keepdim=True)
    B, D_1 = t.shape
    r_k = math.sqrt(D_1)

    result = (r_k / (t_norm_sq + 1)) * torch.cat((2 * t, t_norm_sq - 1), dim=-1)
    
    return result
    
def stereographic_proj_t(x):
    '''
    x: torch.tensor whose size = (n, D)
    '''
    x = x.flatten(1)  # n, D 
    r = (x**2).sum(1, keepdim=True).sqrt()

    return (1/(r - x[:,-1:] + 1e-8)) * x[:,:-1] # n, D-1

def metric_stereographic(t):
    '''
    t:      torch.tensor whose size = (n, D-1)
    out:    torch.tensor whose size = (n, D-1, D-1)
    '''
    t_norm_sq = (t ** 2).sum(1, keepdim=True) # B , 1
    r = (t**2).sum(1, keepdim=True).sqrt() # B, 1

    G = 4 * r ** 4 / ((t_norm_sq + r ** 2) ** 2) * torch.ones_like(t, requires_grad=True) # B, D-1

    return G # B, D-1

def jacobian_of_f(f, z, create_graph=True):
    batch_size, z_dim = z.size()
    # v = torch.eye(z_dim).unsqueeze(0).repeat(batch_size, 1, 1).view(-1, z_dim).to(z)
    v = torch.ones_like(z)
    # z = z.repeat(1, z_dim).view(-1, z_dim)

    out = (
        torch.autograd.functional.jvp(
            f, z, v=v, create_graph=create_graph
        )[1].view(batch_size, z_dim, -1).permute(0, 2, 1)
    )
    return out 

def compose(*functions):
    def inner(arg):
        for f in reversed(functions):
            arg = f(arg)
        return arg
    return inner

def f_h(f):
    def inner(arg):
        return f(arg)[1]
    return inner

def isometry_loss_h(f, x_samples, timesteps, args, device):
    '''
    f:          torch.nn.module class 
    timesteps:  torch.tensor whose size = (n, )
    z_samples:  torch.tensor whose size = (n, 2) # question
    out:        torch.tensor whose size = (1, )
    '''

    bs = x_samples.shape[0]
    u = torch.randn_like(x_samples, device=device)
    Ju = A.jvp(lambda z: f(z, timesteps, return_dict=True)[1], x_samples, u)[1]
    JTJu = A.vjp(lambda z: f(z, timesteps, return_dict=True)[1], x_samples, Ju)[1]

    TrR = torch.sum((Ju.view(bs, -1) ** 2), dim=1).mean()
    TrR2 = torch.sum(JTJu.view(bs, -1) ** 2, dim=1).mean()
    isometry_losses = TrR2 / TrR ** 2 #* torch.sqrt(detG)
    iso_loss = isometry_losses

    return iso_loss

def isometry_loss_t(f, x_samples, timesteps, args, device):
    '''
    f:          torch.nn.module class 
    timesteps:  torch.tensor whose size = (n, )
    z_samples:  torch.tensor whose size = (n, 2) # question
    out:        torch.tensor whose size = (1, )
    '''

    bs = x_samples.shape[0]
    z_samples = stereographic_proj_t(x_samples) # B, D-1
    G = metric_stereographic(z_samples) # B, D-1
    
    u = torch.randn_like(z_samples, device=device)
    J_Ginv_u = A.jvp(lambda z: f(inv_stereographic_proj_t(z).reshape(x_samples.shape), timesteps, return_dict=True)[1], z_samples, 1/torch.sqrt(G) * u, create_graph=True)[1]
    J_Ginv_u_ = A.jvp(lambda z: f(inv_stereographic_proj_t(z).reshape(x_samples.shape), timesteps, return_dict=True)[1], z_samples, 1/G * u, create_graph=True)[1]
    JTJ_Ginv_u = A.vjp(lambda z: f(inv_stereographic_proj_t(z).reshape(x_samples.shape), timesteps, return_dict=True)[1], z_samples, J_Ginv_u_, create_graph=True)[1]

    TrR = torch.sum((J_Ginv_u.view(bs, -1) ** 2), dim=1).mean()
    TrR2 = torch.sum(JTJ_Ginv_u.view(bs, -1) ** 2, dim=1).mean()
    isometry_losses = TrR2 / TrR ** 2 #* torch.sqrt(detG)
    iso_loss = isometry_losses


    return iso_loss

class ExponentialMovingAverage:
    def __init__(self, alpha):
        self.alpha = alpha
        self.ema = None

    def update(self, current_value):
        if self.ema is None:
            self.ema = current_value
        else:
            self.ema = self.alpha * current_value + (1 - self.alpha) * self.ema

    def get_ema(self):
        return self.ema
    
def pl_regularizer(f, x_samples, timesteps, args, device, ema):

    y = torch.randn_like(f(x_samples, timesteps, return_dict=True)[1])
    JTy = A.vjp(lambda z: f(z, timesteps, return_dict=True)[1], x_samples, y)[1]
    pl_length = torch.sqrt(torch.sum(JTy ** 2))

    ema.update(pl_length)

    pl_mean = ema.get_ema()
    pl_loss = (pl_length - pl_mean)**2

    return pl_loss