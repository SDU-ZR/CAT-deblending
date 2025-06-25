import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage import img_as_float
def draw_grid_1():
    from skimage.io import imread

    gs = GridSpec(3, 2)
    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.005, top=1, wspace=0.05, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = np.array(img_as_float(imread("./figure_6/select_test_data/0.png")))
    ax1.imshow(img1)
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = np.array(img_as_float(imread("./figure_6/select_test_data/1.png")))
    ax2.imshow(img2)
    ax3 = fig.add_subplot(gs[1, 0])
    img3 = np.array(img_as_float(imread("./figure_6/select_test_data/2.png")))
    ax3.imshow(img3)
    ax4 = fig.add_subplot(gs[1, 1])
    img4 = np.array(img_as_float(imread("./figure_6/select_test_data/3.png")))
    ax4.imshow(img4)
    ax5 = fig.add_subplot(gs[2, 0])
    img5 = np.array(img_as_float(imread("./figure_6/select_test_data/4.png")))
    ax5.imshow(img5)
    ax6 = fig.add_subplot(gs[2, 1])
    img6 = np.array(img_as_float(imread("./figure_6/select_test_data/5.png")))
    ax6.imshow(img6)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.axis('off')
    # plt.tight_layout()
    plt.savefig('./figure_6/1.png')
    plt.show()

draw_grid_1()