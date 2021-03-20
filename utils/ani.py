import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import datetime

# now_date = str(datetime.datetime.now())
fig = plt.figure()
# plt.title(now_date)
ims = []
def animation_generate(img):
    ims_i = []
    im = plt.imshow(img, cmap='gray')
    ims_i.append([im])
    return ims_i

def save_ani(x_list, filename='v.gif', fps=60):
    ims = []
    fig = plt.figure()
    for img in x_list:
        ims += animation_generate(img)
    ani = animation.ArtistAnimation(fig, ims)
    ani.save(filename, fps=fps, writer='ffmpeg')#'imagemagick')

if __name__ == '__main__':
    ani = animation.ArtistAnimation(fig, ims)
    ani.save("v.gif", writer='imagemagick')
