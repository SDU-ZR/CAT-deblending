import os
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.util import img_as_float


def draw_catalog_1():
    fig = plt.figure()
    # ssim_x = np.around(ssim_x, decimals=2)
    # ssim_y = np.around(ssim_y, decimals=2)
    data = np.load("./figure_9/catalog_deblend_real_galaxy.npy")
    files = np.load("./figure_9/catalog_blend_galaxy_files.npy")

    floders = os.listdir("./figure_9/DRAW")
    df = pd.read_csv("./figure_9/important.csv")
    for floder in floders:
        for png in os.listdir(os.path.join("./figure_9/DRAW",floder)):
            indices = np.where(files == png)[0]
            found_row = df[df["png"] == "{}".format(png)]
            gs = GridSpec(1, 2)
            fig.set_size_inches(12.5, 6)
            fig.subplots_adjust(left=0.00, right=1, bottom=0.00, top=1, wspace=0, hspace=0.0)
            # print(found_row)
            ra = found_row["ra"].values[0]
            dec = found_row["dec"].values[0]

            img1 = np.squeeze(data[indices,0,:,:,:])
            img2 = np.squeeze(data[indices, 1, :, :, :])
            # ax1 = fig.add_subplot(gs[0, 0])
            # ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[0, 0])
            ax4 = fig.add_subplot(gs[0, 1])

            for ax in [ax3, ax4]:
                ax.axis('off')
            # ax1.imshow(up_label)
            # ax1.text(3., 10., r'Preblended 1', color='#FFFFFF')
            # ax2.imshow(down_label)
            # ax2.text(3., 10., r'Preblended 2', color='#FFFFFF')
            ax3.imshow(img1)
            ax3.text(3, 10, r'Blended', color='#FFFFFF',fontsize=32)
            ax3.text(4.3, 120, '{:.2f}+{:.2f}'.format(ra,dec), color='#FFFFFF',fontsize=32)
            ax4.imshow(img2)
            ax4.text(3., 10., r'Deblended', color='#FFFFFF',fontsize=32)
            ax4.text(4.3, 120, '{:.2f}+{:.2f}'.format(ra,dec), color='#FFFFFF',fontsize=32)

            # plt.tight_layout(pad=0)
            plt.subplots_adjust(wspace=0.06, hspace=-0.42)

            if not os.path.exists(os.path.join("./figure_9/figure_catalog/",floder)):
                os.makedirs(os.path.join("./figure_9/figure_catalog",floder))
            # plt.show()
            # break
            fig.savefig("./figure_9/figure_catalog/{}/{}".format(floder,png))
def draw_catalog_2():
    # 循环绘制每个小文件夹中的图片
    from skimage.io import imread
    row = -1
    col = -1
    plt.figure(figsize=(18, 24))
    gs = GridSpec(8, 3)
    floders = os.listdir("./figure_9/figure_catalog")
    random.shuffle(floders)
    print(floders)
    for floder in floders:

        # plt.text(10, 10, floder, fontsize=12, ha='center', va='center')
        row += 1
        for png in os.listdir(os.path.join("./figure_9/figure_catalog",floder)):
            col += 1

            image = img_as_float(imread(os.path.join("./figure_9/figure_catalog",floder,png)))

            # 计算当前子图的位置
            ax = plt.subplot(gs[row, col])
            # 绘制图片
            plt.imshow(image)
            plt.axis('off')

        col = -1
    # 调整子图之间的间距
    # plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, bottom=0.000, top=1, wspace=0.1, hspace=0.0)
    plt.savefig("./figure_9/catalog.png",dpi=300)
    # 展示大图
    plt.show()

def draw_catalog():
    from PIL import Image, ImageDraw, ImageFont

    # 读入图片
    image_path = "./figure_9/catalog.png"  # 将路径替换为你的图片路径
    image = Image.open(image_path)

    # 获取图片宽高
    image_width, image_height = image.size

    # 创建画布对象
    canvas = Image.new('RGB', (image_width + 280, image_height), color='white')

    # 将原图绘制在画布的右侧
    canvas.paste(image, (280, 0))
    # 创建画笔和字体对象
    draw = ImageDraw.Draw(canvas)
    font_size = 140  # 调整字体大小
    font = ImageFont.truetype("arial.ttf", font_size)


    # 添加六行数字
    folder_names = ["E", 'F', 'SE', 'PHI', 'S', 'B','X','Q']
    text_offset = 400

    for i, folder_name in enumerate(folder_names):
        text_position = (30, text_offset + i * 900)
        draw.text(text_position, folder_name, fill='black', font=font)

    # 保存或展示最终图片
    canvas.save('./9.png')  # 如果想保存图片，将路径替换为输出路径
    canvas.show()

if __name__ == '__main__':
    draw_catalog_1()
    draw_catalog_2()
    draw_catalog()