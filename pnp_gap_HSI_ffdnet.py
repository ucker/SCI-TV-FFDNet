import torch
import torch.optim as optim
import  torch.nn as nn
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import cvxpy as cp
import time
import argparse

from utils.utils import psnr, ssim, clip
from utils.ani import save_ani
from model.network_ffdnet import FFDNet

parser = argparse.ArgumentParser(description='Select device')
parser.add_argument('--device', default=0)
# parser.add_argument('--level', default=0)
args = parser.parse_args()
device_num = args.device
# level = float(args.level)
device = 'cuda:{}'.format(device_num)
torch.no_grad()
model = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R').to(device)
model.load_state_dict(torch.load('pretrained_models/ffdnet_gray.pth'))
model.eval()

def ffdnet_denosing(x, sigma, flag):
    image_m, image_n, image_c = x.shape
    if flag:
        x_min = x.min().item()
        x_max = x.max().item()
        scale = 0.7
        shift = (1 - scale) / 2
        x = (x - x_min) / (x_max - x_min)
        x = x * scale + shift
        sigma = torch.tensor(sigma / (x_max - x_min) * scale, device=device)
    else:
        sigma = torch.tensor(sigma, device=device)

    frame_list = []
    with torch.no_grad():
        for j in range(image_c):
            temp_x = x[:, :, j].view(1, 1, image_m, image_n)
            estimate_img = model(temp_x, sigma.view(1, 1, 1, 1))
            frame_list.append(estimate_img[0, 0, :, :])
        x = torch.stack(frame_list, dim=2)

    if flag:
        x = (x - shift) / scale
        x = x * (x_max - x_min) + x_min
    return x

def run():
    image_seq = []
    mat_data = sio.loadmat('./data/toy31_cassi.mat')
    im_orig = mat_data['orig'] / 255.
    im_orig = torch.from_numpy(im_orig).type(torch.float32).to(device)
    image_m, image_n, image_c = im_orig.shape
    mask = torch.from_numpy(mat_data['mask'].astype(np.float32)).to(device)
    shape = im_orig.shape
    y = mat_data['meas'] / 255.
    y = torch.from_numpy(y).type(torch.float32).to(device)
    # data missing and noise
    # y = y + level * torch.randn_like(y)
    # index_rand = np.random.rand(*list(y.shape))
    # index_y = np.argwhere(index_rand < 0.05)
    # y[index_y[:,0], index_y[:,1]] = 0
    x = y.unsqueeze(2).expand_as(mask) * mask
    mask_sum = torch.sum(mask**2, dim=2)
    mask_sum[mask_sum == 0] = 1
    flag = True
    y1 = torch.zeros_like(y, dtype=torch.float32, device=device)
    sigma = 5. / 255
    for i in tqdm(range(100)):
        if i == 20: flag = False
        if i < 20: sigma = 50./255
        elif i < 30: sigma = 25./255
        elif i < 40: sigma = 12./255
        else: sigma = 6./255
        yb = torch.sum(mask * x, dim=2)
        # no Acceleration
        # temp = (y - yb) / mask_sum
        # x = x + 1 * (temp.unsqueeze(2).expand_as(mask) * mask)
        y1 = y1 + (y - yb)
        temp = (y1 - yb) / mask_sum
        x = x + 1 * (temp.unsqueeze(2).expand_as(mask) * mask)

        x = ffdnet_denosing(x, sigma, flag).clamp(0, 1)
        image_seq.append(x[:,:,0].clamp(0., 1.).cpu().numpy())

    # save_ani(image_res, filename='ffd_HSI.mp4', fps=10)
    x.clamp_(0., 1.)
    psnr_ = [psnr(x[..., kv], im_orig[..., kv]) for kv in range(image_c)]
    ssim_ = [ssim(x[..., kv], im_orig[..., kv]) for kv in range(image_c)]
    return np.mean(psnr_), np.mean(ssim_)

begin_time = time.time()
psnr_res, ssim_res = run()
end_time = time.time()
runing_time = end_time - begin_time
print('{:.2f}, {:.4f}, {:.2f}'.format(psnr_res, ssim_res, runing_time))