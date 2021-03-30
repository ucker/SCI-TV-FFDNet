import torch
import os
import torch.optim as optim
import  torch.nn as nn
import numpy as np
import scipy.io as sio
import argparse
from tqdm import tqdm
import cvxpy as cp
import time

from model.TV_denoising import TV_denoising, TV_denoising3d
from utils.utils import clip, ssim, psnr
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
    mat_data = sio.loadmat('./data/toy31_cassi.mat')
    im_orig = mat_data['orig'] / 255
    im_orig = torch.from_numpy(im_orig).type(torch.float32).to(device)
    image_m, image_n, image_c = im_orig.shape
    # image_seq = []
    # ---- load mask matrix ----
    mask = torch.from_numpy(mat_data['mask'].astype(np.float32)).to(device)
    y = mat_data['meas'] / 255
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
    sigma_ = 50 / 255
    for i in tqdm(range(100)):
        if i == 20: flag = False
        yb = torch.sum(mask * x, dim=2)
        # no Acceleration
        # temp = (y - yb) / (mask_sum)
        # x = x + 1 * (temp.unsqueeze(2).expand_as(mask) * mask)
        y1 = y1 + (y - yb)
        temp = (y1 - yb) / mask_sum
        x = x + 1 * (temp.unsqueeze(2).expand_as(mask) * mask)

        if i < 20:
            x = ffdnet_denosing(x, 50./255, flag)
        else:
            ffdnet_hypara_list = [100., 80., 60., 40., 20., 10., 5.]
            ffdnet_num = len(ffdnet_hypara_list)
            tv_hypara_list = [10, 0.01]
            tv_num = len(tv_hypara_list)
            ffdnet_list = [ffdnet_denosing(x, level/255., flag).clamp(0, 1) for level in ffdnet_hypara_list]
            tv_list = [TV_denoising(x, level, 5).clamp(0, 1) for level in tv_hypara_list]

            ffdnet_mat = np.stack(
                [x_ele[:, :, :].cpu().numpy().reshape(-1).astype(np.float64) for x_ele in ffdnet_list],
                axis=0)
            tv_mat = np.stack(
                [x_ele[:, :, :].cpu().numpy().reshape(-1).astype(np.float64) for x_ele in tv_list],
                axis=0)
            w = cp.Variable(ffdnet_num + tv_num)
            P = np.zeros((ffdnet_num + tv_num, ffdnet_num + tv_num))
            P[:ffdnet_num, :ffdnet_num] = ffdnet_mat @ ffdnet_mat.T
            P[:ffdnet_num, ffdnet_num:] = -ffdnet_mat @ tv_mat.T
            P[ffdnet_num:, :ffdnet_num] = -tv_mat @ ffdnet_mat.T
            P[ffdnet_num:, ffdnet_num:] = tv_mat @ tv_mat.T
            one_vector_ffdnet = np.ones((1, ffdnet_num))
            one_vector_tv = np.ones((1, tv_num))
            objective = cp.quad_form(w, P)
            problem = cp.Problem(
                cp.Minimize(objective),
                [one_vector_ffdnet @ w[:ffdnet_num] == 1,
                    one_vector_tv @ w[ffdnet_num:] == 1,
                    w >= 0])
            problem.solve()
            w_value = w.value
            x_ffdnet, x_tv = 0, 0
            for idx in range(ffdnet_num):
                x_ffdnet += w_value[idx] * ffdnet_list[idx]
            for idx in range(tv_num):
                x_tv += w_value[idx + ffdnet_num] * tv_list[idx]
            x = 0.5 * (x_ffdnet + x_tv)
            # image_seq.append(x[...,0])

    x.clamp_(0, 1)
    # fps = 10
    # save_ani(image_seq, filename='HSI.mp4', fps=fps)
    psnr_ = [psnr(x[..., kv], im_orig[..., kv]) for kv in range(image_c)]
    ssim_ = [ssim(x[..., kv], im_orig[..., kv]) for kv in range(image_c)]
    return np.mean(psnr_), np.mean(ssim_)

begin_time = time.time()
psnr_res, ssim_res = run()
end_time = time.time()
running_time = end_time - begin_time
print('{:.2f}, {:.4f}, {:.2f}'.format(psnr_res, ssim_res, running_time))
