import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage import img_as_float
def draw_grid_2():
    from skimage.io import imread

    gs = GridSpec(2, 1)
    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.005, top=1, wspace=0.05, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = np.array(img_as_float(imread("./figure_5/psnr_ssim.png")))
    ax1.imshow(img1)
    ax2 = fig.add_subplot(gs[1, 0])
    img2 = np.array(img_as_float(imread("./figure_5/3_4.png")))
    ax2.imshow(img2)

    for ax in [ax1, ax2]:
        ax.axis('off')
    # plt.tight_layout()
    plt.savefig('./figure_5/5.png')
    plt.show()
def draw_grid_0():
    from skimage.io import imread

    gs = GridSpec(1, 2)
    fig = plt.figure(figsize=(16, 6))
    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.000, top=1, wspace=0.00, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = np.array(img_as_float(imread("./figure_5/3.png")))
    ax1.imshow(img1)
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = np.array(img_as_float(imread("./figure_5/4.png")))
    ax2.imshow(img2)

    for ax in [ax1, ax2]:
        ax.axis('off')
    # plt.tight_layout()
    plt.savefig('./figure_5/3_4.png',dpi=300)
    plt.show()

def draw_grid_1():
    from skimage.io import imread

    gs = GridSpec(1, 2)
    fig = plt.figure(figsize=(16, 6))
    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.000, top=1, wspace=0.00, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = np.array(img_as_float(imread("./figure_5/psnr.png")))
    ax1.imshow(img1)
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = np.array(img_as_float(imread("./figure_5/ssim.png")))
    ax2.imshow(img2)

    for ax in [ax1, ax2]:
        ax.axis('off')
    # plt.tight_layout()
    plt.savefig('./figure_5/psnr_ssim.png',dpi=300)
    plt.show()

if __name__ == '__main__':
    draw_grid_0()
    draw_grid_1()
    draw_grid_2()
