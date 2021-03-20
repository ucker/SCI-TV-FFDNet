import scipy.io as sio
import cv2
import os
import numpy as np

base_dir = '../data/hsi/'
dataset_name = os.listdir(base_dir)
for dataset in dataset_name:
    dataset_path = '{}{}/{}/'.format(base_dir, dataset, dataset)
    print(dataset_path)
    data_name = os.listdir(dataset_path)
    data_name = sorted([n for n in data_name if 'RGB' not in n and 'png' in n])
    hsi = []
    for name in data_name:
        img_path = '{}{}'.format(dataset_path, name)
        temp_img = cv2.imread(img_path, 0)
        hsi.append(temp_img)
    hsi = np.stack(hsi, axis=2)
    print(hsi.shape)
    mask = np.random.uniform(size=hsi.shape)
    mask[mask > 0.5] = 1.
    mask[mask < 0.5] = 0.
    meas = np.sum(mask * hsi, axis=2)
    sio.savemat(
        '../Demo_mat/snapshot_image/CAVE/{}.mat'.format(dataset),
        {'orig': hsi, 'meas': meas, 'mask': mask}
    )
