import scipy.io as sio
import numpy as np
from utils import FILE_NAME


for fn in FILE_NAME:
    testing_data = sio.loadmat('../Demo_mat/snapshot_image/CAVE/{}'.format(fn))['orig']
    print(fn)
    image_m, image_n, image_c = testing_data.shape
    raw_mask_size = (image_m+image_c, image_n)
    print(raw_mask_size)
    raw_masks = np.random.randint(low=0, high=2, size=raw_mask_size)
    masks = np.zeros_like(testing_data, dtype=np.float32)
    for c in range(image_c):
        masks[:,:,c] = raw_masks[c:c+image_m, :]
    measurement_test = np.sum(testing_data * masks, axis=2)
    sio.savemat(
        '../Demo_mat/snapshot_image/CAVE/{}'.format(fn),
        {
            'orig': testing_data,
            'mask': masks,
            'meas': measurement_test
        })
