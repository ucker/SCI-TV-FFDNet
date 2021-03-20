import scipy.io as sio
import numpy as np
from utils import FILE_NAME
from tqdm import tqdm

mask_list = []
for index in tqdm(range(1)):
    image_m, image_n, image_c = (256, 256, 24)
    raw_mask_size = (image_m+image_c, image_n)
    # print(raw_mask_size)
    raw_masks = np.random.randint(low=0, high=2, size=raw_mask_size)
    masks = np.zeros((256, 256, 24), dtype=np.float32)
    for c in range(image_c):
        masks[:,:,c] = raw_masks[c:c+image_m, :]
    mask_list.append(masks)
mask_list = np.array(mask_list)
sio.savemat(
    './Sensing.mat',
    {
        # 'mask': mask_list
        'mask': masks
    })
