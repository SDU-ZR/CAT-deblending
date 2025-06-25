import os

import numpy as np
from matplotlib import pyplot as plt
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

def background_subtraction(image):
    # 使用sigma剪切统计获取图像的均值，中位数和标准差
    import numpy as np
    from astropy.stats import sigma_clipped_stats
    from photutils import datasets, Background2D, MedianBackground

    mean, median, std = sigma_clipped_stats(image, sigma=3.0)

    # 定义一个块大小，块大小应该大于源的大小，并能够包含足够多的背景像素以提供好的统计
    box_size = (20, 20)  # 块大小为50x50像素

    # 创建一个背景对象，使用中位数估计背景
    bkg_estimator = MedianBackground()

    # 创建Background2D对象，包含计算的背景和背景噪声图像
    bkg = Background2D(image, box_size, filter_size=(3, 3), bkg_estimator=bkg_estimator)

    # 提取背景图像
    background_image = bkg.background

    # 打印属于背景的值的平均值
    print('平均背景值: ', np.mean(background_image))
    return background_image
def residual_cal(begin,end,cls):
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    x, y = np.meshgrid(range(128), range(128))


    data = np.load("./val_fake_up_img.npy",mmap_mode='r')
    up_label = np.squeeze(data[begin:end,0,1,:,:]).clip(0, 1)
    fake_up_img = np.squeeze(data[begin:end,2,1,:,:]).clip(0, 1)
    residual_cal = np.squeeze(up_label - fake_up_img)
    mean_array = np.mean(residual_cal, axis=0)
    mea = background_subtraction(mean_array)
    print(np.max(mean_array))
    norm = asin_stretch_norm(mean_array)
    # plt.subplots_adjust(left=0.00, bottom=0.02, right=1, top=0.98, wspace=0, hspace=0)
    # plt.text(3., 10., "'r' band residual mean", color='#070c13',fontsize=12)
    # # 设置坐标轴为紧凑模式，确保图像没有多余的空白
    # plt.axis('tight')

    # 保存图像，设置bbox_inches='tight'确保保存时没有多余的空白
    surf = ax.plot_surface(x, y, mean_array, cmap='viridis', alpha=0.7, norm=norm)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_zlabel('Pixel Value')
    plt.title(' Pixel Distribution of r-band residuals\n in all blended samples ',ha='center')

    ax.view_init(elev=10, azim=-35)
    z = np.zeros_like(mean_array)
    z[:] = mea
    ax.plot_surface(x, y, mea, color='blue', alpha=0.4)
    k = np.zeros_like(mean_array)
    ax.plot_surface(x, y, k, color='r', alpha=0.3)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.6, axes_class=plt.Axes)

    # 添加颜色条
    fig.colorbar(surf,cax=cax)
    plt.subplots_adjust(left=0, right=0.9, top=0.9, bottom=0.05)
    plt.savefig("./figure_7/{}.png".format(cls), dpi=600)
    plt.show()

def draw_grid_6(name_array):
    from skimage.io import imread

    gs = GridSpec(2, 2)
    fig = plt.figure(figsize=(18, 18))
    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.005, top=1, wspace=0, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = np.array(img_as_float(imread("./figure_7/{}.png".format(name_array[0]))))
    ax1.imshow(img1)
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = np.array(img_as_float(imread("./figure_7/{}.png".format(name_array[1]))))
    ax2.imshow(img2)
    ax3 = fig.add_subplot(gs[1, 0])
    img3 = np.array(img_as_float(imread("./figure_7/{}.png".format(name_array[2]))))
    ax3.imshow(img3)
    ax4 = fig.add_subplot(gs[1, 1])
    img4 = np.array(img_as_float(imread("./figure_7/{}.png".format(name_array[3]))))
    ax4.imshow(img4)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.axis('off')
    # plt.tight_layout()
    plt.savefig('./figure_7/7.png')
    plt.show()

if __name__ == "__main__":
    from matplotlib.gridspec import GridSpec
    from skimage import img_as_float
    for begin, end, name in zip(
            [0, 4000, 8000, 0],
            [4000, 8000, 12000, 12000],
            [
                "Pixel Distribution of r-band residuals in two blended samples",
                "Pixel Distribution of r-band residuals in three blended samples",
                "Pixel Distribution of r-band residuals in four blended samples",
                "residual_r_band_12000_3d.png"
            ]
    ):
        residual_cal(begin, end, name)

    draw_grid_6([
                "Pixel Distribution of r-band residuals in two blended samples",
                "Pixel Distribution of r-band residuals in three blended samples",
                "Pixel Distribution of r-band residuals in four blended samples",
                "residual_r_band_12000_3d.png"
            ])








