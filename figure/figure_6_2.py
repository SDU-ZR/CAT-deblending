import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage import img_as_float
DIST_BINS = [(2,8), (8, 13), (13, 18), (18, 60)]
MAGDIFF_BINS = [(0, 0.35), (0.35, 0.76), (0.76, 1.25), (1.25, 2)]
import matplotlib.colors as mcl

paper_palette = [
    '#c6bcc0',
    '#1d456d',
    '#759792',
    '#ba9123',
    '#2f6b99',
    '#64532e',
    '#070c13',
    '#a3351a',
    '#0f3849',
    '#c66978',
    '#d5b56b',
    '#19252e',
    '#111b24',
    '#2a5650',
    '#24352b',
    '#162423',
    '#0f1c1b',
    '#1c181e',
    '#34241c',
]
img_cmap = mcl.LinearSegmentedColormap.from_list('paper_BlBr', ['#1d456d', (0.8,0.8,0.8), '#64532e'])
from astropy.visualization import ImageNormalize
from astropy.visualization import MinMaxInterval
from astropy.visualization import AsinhStretch


def asin_stretch_norm(images):
    return ImageNormalize(
        images,
        interval=MinMaxInterval(),
        stretch=AsinhStretch(),
    )


def find_index(arr, target):
    index = np.where(arr == target)[0]
    return index if index.size > 0 else -1


def plot_blends(blends, indices,k):
    norm = asin_stretch_norm(blends[indices])
    df = pd.read_csv("./test.csv")
    mag_err = df.loc[k, "mag_err"]
    psnr = df.loc[k, "S/N"]
    gs = GridSpec(1, 3)
    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.005, top=0.9, wspace=0, hspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = blends[0]
    ax1.imshow(img1, norm=norm, cmap=img_cmap)
    ax1.text(3, 10, r'Blended', color='#FFFFFF')
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = blends[1]
    ax2.imshow(img2, norm=norm, cmap=img_cmap)
    ax2.text(3., 10., r'Deblended', color='#FFFFFF')
    ax2.text(3., 120., "mag_err :" + str(round(mag_err, 4)), color='#FFFFFF')  #
    ax2.text(70., 120., "psnr :" + str(round(psnr, 2)) + " dB", color='#FFFFFF')  #
    ax3 = fig.add_subplot(gs[0, 2])
    img3 = blends[2]
    ax3.imshow(img3, norm=norm, cmap=img_cmap)
    ax3.text(3., 10., r'Preblend', color='#FFFFFF')
    # ax4 = fig.add_subplot(gs[0, 3])
    # ax4.imshow(mask[2,:,:])

    fig.text(0.16, 0.95, "input Deblender", ha="center", va="center", fontsize=12, color='black')
    fig.text(0.5, 0.95, "output Deblender", ha="center", va="center", fontsize=12,
             color='black')
    fig.text(0.84, 0.95, "target", ha="center", va="center", fontsize=12, color='black')
    # fig.text(0.87, 0.95, "'z' band mask", ha="center", va="center", fontsize=16, color='black')

    for ax in [ax1, ax2, ax3]:
        ax.axis('off')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0.06, hspace=-0.42)
    plt.savefig('./figure_6/band/{}.png'.format(k))
    plt.show()




def main(k):
    blends_images = np.load("./val_fake_up_img.npy",
                            mmap_mode='r')
    print(blends_images.shape)

    muti_band_image = np.zeros((3, 128, 128))

    l = 0
    # For each distance/mag_diff bin, randomly select one blend in the catalog
    blend_indices = range(3)

    muti_band_image[l, :, :] = blends_images[k, 1, 1, :, :] * 2
    l += 1
    muti_band_image[l, :, :] = blends_images[k, 2, 1, :, :] * 2
    l += 1
    muti_band_image[l, :, :] = blends_images[k, 0, 1, :, :] * 2

    plot_blends(muti_band_image, blend_indices,k)
def draw_grid_2():
    from skimage.io import imread

    gs = GridSpec(3, 2)
    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.005, top=1, wspace=0.05, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = np.array(img_as_float(imread("./figure_6/band/7.png")))
    ax1.imshow(img1)
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = np.array(img_as_float(imread("./figure_6/band/2000.png")))
    ax2.imshow(img2)
    ax3 = fig.add_subplot(gs[1, 0])
    img3 = np.array(img_as_float(imread("./figure_6/band/6004.png")))
    ax3.imshow(img3)
    ax4 = fig.add_subplot(gs[1, 1])
    img4 = np.array(img_as_float(imread("./figure_6/band/10001.png")))
    ax4.imshow(img4)
    ax5 = fig.add_subplot(gs[2, 0])
    img5 = np.array(img_as_float(imread("./figure_6/band/10002.png")))
    ax5.imshow(img5)
    ax6 = fig.add_subplot(gs[2, 1])
    img6 = np.array(img_as_float(imread("./figure_6/band/11180.png")))
    ax6.imshow(img6)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.axis('off')
    # plt.tight_layout()
    plt.savefig('./figure_6/2.png')
    plt.show()

for i in [7,2000,6004,10001,10002,11180]:
    main(i)
draw_grid_2()