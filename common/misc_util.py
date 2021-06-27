import numpy as np
import random
import gym
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import torchvision

def set_global_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_global_log_levels(level):
    gym.logger.set_level(level)


def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def adjust_lr(optimizer, init_lr, timesteps, max_timesteps):
    lr = init_lr * (1 - (timesteps / max_timesteps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def get_n_params(model):
    return str(np.round(np.array([p.numel() for p in model.parameters()]).sum() / 1e6, 3)) + ' M params'

def visualize_attn_softmax(I, c, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    N,C,W,H = c.size()
    a = F.softmax(c.view(N,C,-1), dim=2).view(N,C,W,H)
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = torchvision.utils.make_grid(a, nrow=nrow, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2,0,1)

def visualize_attn(I, c1, c2, c3):
    I = I[:25]
    c1 = c1[:25]
    c2 = c2[:25]
    c3 = c3[:25]

    I = torch.from_numpy(I)
    img = torchvision.utils.make_grid(I, nrow=5, normalize=True, scale_each=True)
    img = img.permute((1,2,0)).cpu().numpy()
    
    N,C,W,H = c1.size()
    a1 = F.softmax(c1.view(N,C,-1), dim=2).view(N,C,W,H)
    a1 = F.interpolate(a1, scale_factor=2, mode='bilinear', align_corners=False)
    attn1 = torchvision.utils.make_grid(a1, nrow=5, normalize=True,scale_each=True)
    attn1 = attn1.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn1 = cv2.applyColorMap(attn1, cv2.COLORMAP_JET)
    attn1 = cv2.cvtColor(attn1, cv2.COLOR_BGR2RGB)
    attn1 = np.float32(attn1) / 255
    vis1 = 0.6 * img + 0.4 * attn1

    N,C,W,H = c2.size()
    a2 = F.softmax(c2.view(N,C,-1), dim=2).view(N,C,W,H)
    a2 = F.interpolate(a2, scale_factor=4, mode='bilinear', align_corners=False)
    attn2 = torchvision.utils.make_grid(a2, nrow=5, normalize=True,scale_each=True)
    attn2 = attn2.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn2 = cv2.applyColorMap(attn2, cv2.COLORMAP_JET)
    attn2 = cv2.cvtColor(attn2, cv2.COLOR_BGR2RGB)
    attn2 = np.float32(attn2) / 255
    vis2 = 0.6 * img + 0.4 * attn2

    N,C,W,H = c3.size()
    a3 = F.softmax(c3.view(N,C,-1), dim=2).view(N,C,W,H)
    a3 = F.interpolate(a3, scale_factor=8, mode='bilinear', align_corners=False)
    attn3 = torchvision.utils.make_grid(a3, nrow=5, normalize=True,scale_each=True)
    attn3 = attn3.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn3 = cv2.applyColorMap(attn3, cv2.COLORMAP_JET)
    attn3 = cv2.cvtColor(attn3, cv2.COLOR_BGR2RGB)
    attn3 = np.float32(attn3) / 255
    vis3 = 0.6 * img + 0.4 * attn3

    img = torch.from_numpy(img).permute(2,0,1)
    vis1 = torch.from_numpy(vis1).permute(2,0,1)
    vis2 = torch.from_numpy(vis2).permute(2,0,1)
    vis3 = torch.from_numpy(vis3).permute(2,0,1)
    out = torchvision.utils.make_grid([img, vis1, vis2,vis3], nrow=1)
    return out