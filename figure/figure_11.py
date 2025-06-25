
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

#----------------------------------------------------------------------------------------------------------------------#
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.stats import binned_statistic_2d


import math
import os

import numpy as np

"""
Transformation from raw image data (nanomaggies) to the rgb values displayed
at the legacy viewer https://www.legacysurvey.org/viewer

Code copied from
https://github.com/legacysurvey/imagine/blob/master/map/views.py
"""


def sdss_rgb(imgs, bands, scales=None,
             m=0.02):
    import numpy as np
    rgbscales = {'u': (2, 1.5),  # 1.0,
                 'g': (2, 2.5),
                 'r': (1, 1.5),
                 'i': (0, 1.0),
                 'z': (0, 0.4),  # 0.3
                 }
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img, band in zip(imgs, bands):
        plane, scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)

    # b,g,r = [rimg * rgbscales[b] for rimg,b in zip(imgs, bands)]
    # r = np.maximum(0, r + m)
    # g = np.maximum(0, g + m)
    # b = np.maximum(0, b + m)
    # I = (r+g+b)/3.
    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H, W = I.shape
    rgb = np.zeros((H, W, 3), np.float32)
    for img, band in zip(imgs, bands):
        plane, scale = rgbscales[band]
        rgb[:, :, plane] = (img * scale + m) * fI / I

    # R = fI * r / I
    # G = fI * g / I
    # B = fI * b / I
    # # maxrgb = reduce(np.maximum, [R,G,B])
    # # J = (maxrgb > 1.)
    # # R[J] = R[J]/maxrgb[J]
    # # G[J] = G[J]/maxrgb[J]
    # # B[J] = B[J]/maxrgb[J]
    # rgb = np.dstack((R,G,B))
    rgb = np.clip(rgb, 0, 1)
    return rgb


def dr2_rgb(rimgs, bands, **ignored):
    return sdss_rgb(rimgs, bands, scales=dict(g=(2, 6.0), r=(1, 3.4), z=(0, 2.2)), m=0.03)
#---------------------------------------------------------------------------------------------------------------------#

def find_index(arr, target):
    index = np.where(arr == target)[0]
    return index if index.size > 0 else -1


def make_segmap(img):
    import sep
    SEXCONFIG = {
        "hot": {
            "final_area": 6,
            "final_threshold": 4,
            "final_cont": 0.0001,
            "final_nthresh": 64,
        },
        "cold": {
            "final_area": 10,
            "final_threshold": 5,
            "final_cont": 0.01,
            "final_nthresh": 64,
        },
        "mine": {
            "final_area": 10,
            "final_threshold": 4,
            "final_cont": 0.0001,
            "final_nthresh": 64,
        }
    }

    def run_sextractor(image, background, config):
        return sep.extract(
            image,
            config["final_threshold"],
            err=background.globalrms,
            minarea=config["final_area"],
            deblend_nthresh=config["final_nthresh"],
            deblend_cont=config["final_cont"],
            segmentation_map=True,
        )

    def analyse_single(fits_loc):
        # Compute the background
        img = fits_loc
        single_source_image = np.squeeze(img)
        bkg = sep.Background(single_source_image)
        # single_source_image = single_source_image - bkg
        # Run detection with the 'cold' strategy
        source, segmap = run_sextractor(single_source_image, bkg, SEXCONFIG["cold"])

        return segmap

    return analyse_single(img)


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


def main():
    blends_catalog = np.load("./figure_9/catalog_png_files.npy")
    print(len(blends_catalog))
    blends_images = np.load(os.path.join("./figure_9/val_fake_up_img_catalog.npy"),
                            mmap_mode='r')
    print(blends_images.shape)

    muti_band_image = np.zeros((36, 128, 128))
    rgb_image = np.zeros((36,128,128,3))

    class_blends = os.listdir("./figure_11/DRAW_2")
    print(len(class_blends))
    data = np.load("./figure_9/catalog_deblend_real_galaxy.npy")
    files = np.load("./figure_9/catalog_blend_galaxy_files.npy")
    l = 0
    s = 0
    from skimage.io import imread
    from skimage.util import img_as_float
    S = [0,0,1,2,1,2]
    df = pd.read_csv("./figure_9/important.csv")

    # For each distance/mag_diff bin, randomly select one blend in the catalog
    blend_indices = range(36)
    a = 0
    ras = []
    decs = []
    for blend in class_blends[0:1]:
        d = 10
        pngs = os.listdir(os.path.join("./figure_11/DRAW_2", blend))
        print(len(pngs))
        # pngs = [pngs[S[a]]]
        a += 1

        for png in pngs:
            d += 1
            indices = np.where(files == png)[0]
            found_row = df[df["png"] == "{}".format(png)]
            ra = found_row["ra"].values[0]
            dec = found_row["dec"].values[0]
            ras.append(ra)
            decs.append(dec)
            k = find_index(blends_catalog, png)
            # image = img_as_float(imread(os.path.join("./figure_catalog", floder, png)))
            # muti_band_image[l, :, :] = np.clip(blends_images[k, 0, 1, :, :],-1,1)
            # l += 1
            # muti_band_image[l, :, :] = blends_images[k, 1, 1, :, :]
            # l += 1

            # rgb_image[s, :, :, :] = img_as_float(imread(
            #     os.path.join("D:/original_data_npy/blend_data/catalog_blend_data_png", "{}".format(k[0]), "img.png")))
            # s += 1
            # rgb_image[s, :, :, :] = img_as_float(imread(
            #     os.path.join("D:/original_data_npy/blend_data/catalog_blend_data_png", "{}".format(k[0]),
            #                  "deblend_img.png")))
            # s += 1
            # rgb_image[s, :, :, :] = img_as_float(imread(
            #     os.path.join("D:/original_data_npy/blend_data/catalog_blend_data_png", "{}".format(k[0]),
            #                  "residual_img.png")))
            # s += 1
            fig = plt.figure()
            fig.set_size_inches(5 * 3, 5 * 2)
            fig.subplots_adjust(left=0.02, right=0.998, bottom=0.02, top=1, wspace=0.1, hspace=0.05)
            ax1 = fig.add_subplot(2, 3, 1)
            ax2 = fig.add_subplot(2, 3, 4)
            # ax7 = fig.add_subplot(3, 3, 7)


            ax3 = fig.add_subplot(2, 3, 2, projection='3d')
            ax4 = fig.add_subplot(2, 3, 5, projection='3d')
            # ax8 = fig.add_subplot(3, 3, 8, projection='3d')
            ax5 = fig.add_subplot(2, 3, 3)
            ax6 = fig.add_subplot(2, 3, 6)
            # ax9 = fig.add_subplot(3, 3, 9)


            ax1.imshow(np.squeeze((np.clip(blends_images[k, 0, 1, :, :],0,1))),norm=asin_stretch_norm(np.squeeze(np.clip(blends_images[k, 0, 1, :, :],0,1))),cmap=img_cmap)
            ax1.text(4.3, 120, '{:.2f}+{:.2f}'.format(ra, dec), color='#FFFFFF', fontsize=28)
            ax1.text(4., 10., "real_blended_band", color='#FFFFFF', fontsize=20)
            ax1.axis('off')
            ax2.imshow(np.squeeze(blends_images[k, 1, 1, :, :]),norm=asin_stretch_norm(np.squeeze(blends_images[k, 1, 1, :, :])),cmap=img_cmap)
            ax2.text(4.3, 120, '{:.2f}+{:.2f}'.format(ra, dec), color='#FFFFFF', fontsize=28)
            ax2.text(4., 10., "Deblended_band", color='#FFFFFF', fontsize=20)
            ax2.axis('off')
            # ax7.imshow(np.squeeze(np.clip(blends_images[k, 0, 1, :, :],0,1) -blends_images[k, 1, 1, :, :]),norm=asin_stretch_norm(np.squeeze(np.clip(blends_images[k, 0, 1, :, :],0,1) -blends_images[k, 1, 1, :, :])),cmap=img_cmap)
            # ax7.text(4.3, 120, '{:.2f}+{:.2f}'.format(ra, dec), color='#FFFFFF', fontsize=28)
            # ax7.text(4., 10., "Residual_band", color='#FFFFFF', fontsize=20)
            # ax7.axis('off')

            x, y = np.meshgrid(range(128), range(128))
            ax3.plot_surface(x, y, np.flipud(np.squeeze(np.clip(blends_images[k, 0, 1, :, :],0,1))), cmap='viridis', alpha=0.7, norm=asin_stretch_norm(np.squeeze(blends_images[k, 0, 1, :, :])))
            ax4.plot_surface(x, y, np.flipud(np.squeeze(blends_images[k, 1, 1, :, :])), cmap='viridis', alpha=0.7, norm=asin_stretch_norm(np.squeeze(blends_images[k, 1, 1, :, :])))
            # ax8.plot_surface(x, y, np.rot90(np.squeeze(np.clip(blends_images[k, 0, 1, :, :],0,1) -blends_images[k, 1, 1, :, :])), cmap='viridis', alpha=0.7, norm=asin_stretch_norm(np.clip(blends_images[k, 0, 1, :, :],0,1) -blends_images[k, 1, 1, :, :]))
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_facecolor('lightgray')
            ax4.set_facecolor('lightgray')
            # ax8.set_ylabel('Y')
            # ax8.set_facecolor('lightgray')

            # ax3.set_box_aspect([1, 1, 1])
            # ax4.set_box_aspect([1, 1, 1])
            # ax3.view_init(elev=10, azim=-35)
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')

            map_1 = make_segmap(blends_images[k, 0, 1, :, :])
            map_2 = make_segmap(blends_images[k, 1, 1, :, :])
            map_3 = make_segmap(make_segmap(np.clip(blends_images[k, 0, 1, :, :],0,1) -blends_images[k, 1, 1, :, :]))
            ax5.imshow(np.flipud(map_1), origin='lower', interpolation='nearest', cmap='tab20')
            ax5.axis('off')

            ax6.imshow(np.flipud(map_2), origin='lower', interpolation='nearest', cmap='tab20')
            # 关闭坐标轴
            ax6.axis('off')

            # ax9.imshow(np.flipud(map_3), origin='lower', interpolation='nearest', cmap='tab20')
            # # 关闭坐标轴
            # ax9.axis('off')

            if not os.path.exists("./figure_11/figure_11_1"):
                os.makedirs("../figure_11_1")

            plt.savefig("./figure_11/figure_11_1/draw_{}.png".format(d))
            plt.show()
            # break
        break

def main_2():
    blends_catalog = np.load("./figure_9/catalog_png_files.npy")
    print(len(blends_catalog))
    blends_images = np.load(os.path.join("./figure_9/val_fake_up_img_catalog.npy"),
                            mmap_mode='r')
    print(blends_images.shape)

    muti_band_image = np.zeros((36, 128, 128))
    rgb_image = np.zeros((36,128,128,3))

    class_blends = os.listdir("./figure_11/DRAW_2")
    print(len(class_blends))
    data = np.load("./figure_9/catalog_deblend_real_galaxy.npy")
    files = np.load("./figure_9/catalog_blend_galaxy_files.npy")
    l = 0
    s = 0
    from skimage.io import imread
    from skimage.util import img_as_float
    S = [0,0,1,2,1,2]
    df = pd.read_csv("./figure_9/important.csv")

    # For each distance/mag_diff bin, randomly select one blend in the catalog
    blend_indices = range(36)
    a = 0
    ras = []
    decs = []
    for blend in class_blends[0:1]:
        pngs = os.listdir(os.path.join("./figure_11/DRAW_2/", blend))
        print(len(pngs))
        # pngs = [pngs[S[a]]]
        a += 1
        d = 30

        for png in pngs:
            d += 1
            indices = np.where(files == png)[0]
            found_row = df[df["png"] == "{}".format(png)]
            ra = found_row["ra"].values[0]
            dec = found_row["dec"].values[0]
            ras.append(ra)
            decs.append(dec)
            k = find_index(blends_catalog, png)
            # image = img_as_float(imread(os.path.join("./figure_catalog", floder, png)))
            # muti_band_image[l, :, :] = np.clip(blends_images[k, 0, 1, :, :],-1,1)
            # l += 1
            # muti_band_image[l, :, :] = blends_images[k, 1, 1, :, :]
            # l += 1

            # rgb_image[s, :, :, :] = img_as_float(imread(
            #     os.path.join("D:/original_data_npy/blend_data/catalog_blend_data_png", "{}".format(k[0]), "img.png")))
            # s += 1
            # rgb_image[s, :, :, :] = img_as_float(imread(
            #     os.path.join("D:/original_data_npy/blend_data/catalog_blend_data_png", "{}".format(k[0]),
            #                  "deblend_img.png")))
            # s += 1
            # rgb_image[s, :, :, :] = img_as_float(imread(
            #     os.path.join("D:/original_data_npy/blend_data/catalog_blend_data_png", "{}".format(k[0]),
            #                  "residual_img.png")))
            # s += 1
            fig = plt.figure()
            fig.set_size_inches(5 * 3, 5 * 2)
            fig.subplots_adjust(left=0.02, right=0.998, bottom=0.02, top=1, wspace=0.1, hspace=0.05)
            ax1 = fig.add_subplot(2, 3, 1)
            ax2 = fig.add_subplot(2, 3, 4)

            ax3 = fig.add_subplot(2, 3, 2, projection='3d')
            ax4 = fig.add_subplot(2, 3, 5, projection='3d')
            ax5 = fig.add_subplot(2, 3, 3)
            ax6 = fig.add_subplot(2, 3, 6)


            image_1 = np.zeros([3,128,128])

            image_2 = np.zeros([3,128,128])
            image_1[0,:,:] = np.clip(np.squeeze(blends_images[k, 0, 0, :, :]),-1,1) * 1
            image_1[1, :, :] = np.clip(np.squeeze(blends_images[k, 0, 1, :, :]), -1, 1) * 2
            image_1[2, :, :] = np.clip(np.squeeze(blends_images[k, 0, 2, :, :]), -1, 1) * 4

            image_2[0, :, :] = np.clip(np.squeeze(blends_images[k, 1, 0, :, :]), -1, 1) * 1
            image_2[1, :, :] = np.clip(np.squeeze(blends_images[k, 1, 1, :, :]), -1, 1) * 2
            image_2[2, :, :] = np.clip(np.squeeze(blends_images[k, 1, 2, :, :]), -1, 1) * 4



            img_1 = dr2_rgb(image_1,['g','r','z'])
            img_2 = dr2_rgb(image_2, ['g', 'r', 'z'])
            ax1.imshow(img_1,interpolation='none')
            ax1.text(4.3, 120, '{:.2f}+{:.2f}'.format(ra, dec), color='#FFFFFF', fontsize=28)
            ax1.text(4., 10., "real_blended_rgb", color='#FFFFFF', fontsize=20)
            ax1.axis('off')

            ax2.imshow(img_2,interpolation='none')
            ax2.text(4.3, 120, '{:.2f}+{:.2f}'.format(ra, dec), color='#FFFFFF', fontsize=28)
            ax2.text(4., 10., "Deblended_rgb", color='#FFFFFF', fontsize=20)


            ax2.axis('off')
            x, y = np.meshgrid(range(128), range(128))
            ax3.plot_surface(x, y, np.rot90(np.squeeze(np.clip(blends_images[k, 0, 1, :, :],0,1))), cmap='viridis', alpha=0.7, norm=asin_stretch_norm(np.squeeze(blends_images[k, 0, 1, :, :])))
            ax4.plot_surface(x, y, np.rot90(np.squeeze(blends_images[k, 1, 1, :, :])), cmap='viridis', alpha=0.7, norm=asin_stretch_norm(np.squeeze(blends_images[k, 1, 1, :, :])))
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_facecolor('lightgray')
            ax4.set_facecolor('lightgray')
            # ax3.set_box_aspect([1, 1, 1])
            # ax4.set_box_aspect([1, 1, 1])
            # ax3.view_init(elev=10, azim=-35)
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')

            map_1 = make_segmap(blends_images[k, 0, 1, :, :])
            map_2 = make_segmap(blends_images[k, 1, 1, :, :])
            ax5.imshow(np.flipud(map_1), origin='lower', interpolation='nearest', cmap='tab20')

            ax5.axis('off')
            ax6.imshow(np.flipud(map_2), origin='lower', interpolation='nearest', cmap='tab20')
            # 关闭坐标轴
            ax6.axis('off')
            if not os.path.exists("./figure_11/figure_11_2"):
                os.makedirs("./figure_11/figure_11_2")
            plt.savefig("./figure_11/figure_11_2/draw_{}.png".format(d))
            plt.show()
            # break

        break
from matplotlib.gridspec import GridSpec
from skimage import img_as_float
def draw_grid():
    from skimage.io import imread
    gs = GridSpec(4, 2)
    fig = plt.figure(figsize=(18, 23))
    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.005, top=1, wspace=0, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = np.array(img_as_float(imread("./figure_11/figure_11_1/draw_11.png")))
    ax1.imshow(img1)
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = np.array(img_as_float(imread("./figure_11/figure_11_1/draw_13.png")))
    ax2.imshow(img2)
    ax3 = fig.add_subplot(gs[1, 0])
    img3 = np.array(img_as_float(imread("./figure_11/figure_11_1/draw_14.png")))
    ax3.imshow(img3)
    ax4 = fig.add_subplot(gs[1, 1])
    img4 = np.array(img_as_float(imread("./figure_11/figure_11_1/draw_24.png")))
    ax4.imshow(img4)
    ax5 = fig.add_subplot(gs[2, 0])
    img5 = np.array(img_as_float(imread("./figure_11/figure_11_2/draw_35.png")))
    ax5.imshow(img5)
    ax6 = fig.add_subplot(gs[2, 1])
    img6 = np.array(img_as_float(imread("./figure_11/figure_11_2/draw_39.png")))
    ax6.imshow(img6)
    ax7 = fig.add_subplot(gs[3, 0])
    img7 = np.array(img_as_float(imread("./figure_11/figure_11_2/draw_41.png")))
    ax7.imshow(img7)
    ax8 = fig.add_subplot(gs[3, 1])
    img8 = np.array(img_as_float(imread("./figure_11/figure_11_2/draw_44.png")))
    ax8.imshow(img8)

    for ax in [ax1,ax2,ax3, ax4,ax5,ax6,ax7,ax8]:
        ax.axis('off')
    # plt.tight_layout()
    plt.savefig('./11.pdf',dpi=50)
    plt.show()

if __name__ == "__main__":
    # main()
    # main_2()
    draw_grid()
