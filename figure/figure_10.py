import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDirectionArrows

from rgb_train.download_images_threaded import dr2_style_rgb


#----------------------------------------------------------------------------------------#
def save_carefully_resized_png(native_image,loc):
    """
    # TODO
    Args:
        png_loc ():
        native_image ():
        target_size ():

    Returns:

    """
    native_pil_image = Image.fromarray(np.uint8(native_image * 255.), mode='RGB')
    nearest_image = native_pil_image.resize(size=(128, 128), resample=Image.LANCZOS)
    nearest_image = nearest_image.transpose(Image.FLIP_TOP_BOTTOM)  # to align with north/east
    nearest_image.save(loc)



def save_png(resize_image):


    _scales = dict(
        g=(2, 0.008),
        r=(1, 0.014),
        z=(0, 0.019))
    _mnmx = (-0.5, 300)

    rgbimg = dr2_style_rgb(
        (resize_image[0, :, :], resize_image[1, :, :], resize_image[2, :, :]),
        'grz',
        mnmx=_mnmx,
        arcsinh=1.,
        scales=_scales,
        desaturate=True)
    # native_pil_image = Image.fromarray(np.uint8(rgbimg * 255.), mode='RGB')
    # nearest_image = native_pil_image.resize(size=(128, 128), resample=Image.LANCZOS)
    return rgbimg

#--------------------------------------------------------------------------------------------#


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


def plot_blends(blends,rgb_blends,indices, ras,decs,log):
    fig, axes = plt.subplots(nrows=8, ncols=6, sharex=True, sharey=True, figsize=(30, 38), tight_layout=True)


    txts = ["real_blended_band","Deblended_band","Deblended_band residual"]
    txts_2 = ["real_blended_rgb","Deblended_rgb","Deblended_rgb residual"]
    d = 0
    s = 0
    k = 1
    o = 0
    for i, ax in enumerate(axes.flatten()):
        if i < 24:
            idx = indices[i]
            norm = asin_stretch_norm(blends[idx])
            ax.imshow(blends[idx], norm=norm, cmap=img_cmap)
            ax.text(4., 10., r'{}_{}'.format(txts[d], k), color='#FFFFFF', fontsize=20)
            print(ras[o])
            print(decs[o])
            ax.text(4.3, 120, '{:.2f}+{:.2f}'.format(ras[o],decs[o]), color='#FFFFFF',fontsize=32)
            d += 1
            if d == 3:
                o += 1
                d = 0
                k += 1
            # ax.set_title(f"ID={idx}, dist={small_cat.distance[idx]:.2f}, magdiff={small_cat.magdiff[idx]:.2f}")
            ax.set_axis_off()
        else:
            if i == 24:
                k = 1
                o = 0

            idx = indices[i]
            ax.imshow(rgb_blends[idx-24])
            ax.text(4., 10., r'{}_{}'.format(txts_2[s], k), color='#FFFFFF', fontsize=20)
            ax.text(4.3, 120, '{:.2f}+{:.2f}'.format(ras[o], decs[o]), color='#FFFFFF', fontsize=32)
            # ax.set_title(f"ID={idx}, dist={small_cat.distance[idx]:.2f}, magdiff={small_cat.magdiff[idx]:.2f}")
            s += 1
            if s == 3:
                s = 0
                o += 1
                k += 1
            ax.set_axis_off()



    # a = AnchoredDirectionArrows(
    #     axes[0, 0].transAxes, r'deblend', 'class', loc='upper left',
    #     aspect_ratio=-1,
    #     sep_x=0.02, sep_y=-0.04,
    #     color='white'
    # )
    # axes[0, 0].add_artist(a)

    fig.savefig(f"./figure_10/100.png",dpi=100)
    plt.show()

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


def main():
    blends_catalog = np.load("./figure_9/catalog_png_files.npy")
    blends_images = np.load("./figure_9/val_fake_up_img_catalog.npy",
                            mmap_mode='r')
    print(blends_images.shape)

    muti_band_image = np.zeros((48, 128, 128))
    rgb_image = np.zeros((48,128,128,3))

    class_blends = os.listdir("./figure_9/DRAW/")
    data = np.load("./figure_9/catalog_deblend_real_galaxy.npy")
    files = np.load("./figure_9/catalog_blend_galaxy_files.npy")
    l = 0
    s = 0
    from skimage.io import imread
    from skimage.util import img_as_float
    S = [0,0,1,2,1,2,1,2]
    df = pd.read_csv("./figure_9/important.csv")

    # For each distance/mag_diff bin, randomly select one blend in the catalog
    blend_indices = range(48)
    a = 0
    ras = []
    decs = []
    for blend in class_blends:

        pngs = os.listdir(os.path.join("./figure_9/DRAW/", blend))
        pngs = [pngs[S[a]]]
        a += 1

        for png in pngs:
            indices = np.where(files == png)[0]
            found_row = df[df["png"] == "{}".format(png)]
            ra = found_row["ra"].values[0]
            dec = found_row["dec"].values[0]
            ras.append(ra)
            decs.append(dec)
            k = find_index(blends_catalog, png)
            # image = img_as_float(imread(os.path.join("./figure_catalog", floder, png)))
            muti_band_image[l, :, :] = np.clip(blends_images[k, 0, 1, :, :],-1,1)
            l += 1
            muti_band_image[l, :, :] = blends_images[k, 1, 1, :, :]
            l += 1
            muti_band_image[l, :, :] = np.clip(blends_images[k, 0, 1, :, :],0,1) -blends_images[k, 1, 1, :, :]
            l += 1
            image_1 = np.zeros([3,128,128])
            image_2 = np.zeros([3,128,128])
            image_3 = np.zeros([3,128,128])
            image_1[0,:,:] = np.clip(np.squeeze(blends_images[k, 0, 0, :, :]),-1,1) * 1
            image_1[1, :, :] = np.clip(np.squeeze(blends_images[k, 0, 1, :, :]), -1, 1) * 2
            image_1[2, :, :] = np.clip(np.squeeze(blends_images[k, 0, 2, :, :]), -1, 1) * 4
            image_2[0, :, :] = np.clip(np.squeeze(blends_images[k, 1, 0, :, :]), -1, 1) * 1
            image_2[1, :, :] = np.clip(np.squeeze(blends_images[k, 1, 1, :, :]), -1, 1) * 2
            image_2[2, :, :] = np.clip(np.squeeze(blends_images[k, 1, 2, :, :]), -1, 1) * 4
            image_3[0, :, :] = np.clip(np.squeeze(blends_images[k, 0, 0, :, :] -blends_images[k, 1, 0, :, :]), -1, 1) * 1
            image_3[1, :, :] = np.clip(np.squeeze(blends_images[k, 0, 1, :, :] -blends_images[k, 1, 1, :, :]), -1, 1) * 2
            image_3[2, :, :] = np.clip(np.squeeze(blends_images[k, 0, 2, :, :] -blends_images[k, 1, 2, :, :]), -1, 1) * 4


            # rgb_image[s, :, :, :] = dr2_rgb(image_1,['g','r','z'])
            # s += 1
            # rgb_image[s, :, :, :] = dr2_rgb(image_2,['g','r','z'])
            # s += 1
            # rgb_image[s, :, :, :] = dr2_rgb(image_3,['g','r','z'])
            # s += 1


            rgb_image[s, :, :, :] = save_png(image_1)
            s += 1
            rgb_image[s, :, :, :] = save_png(image_2)
            s += 1
            rgb_image[s, :, :, :] = save_png(image_3)
            s += 1

    plot_blends(muti_band_image, rgb_image,blend_indices, ras,decs,log=True)



def draw_catalog():
    from PIL import Image, ImageDraw, ImageFont

    # 读入图片
    image_path = "./figure_10/100.png"  # 将路径替换为你的图片路径
    image = Image.open(image_path)

    # 获取图片宽高
    image_width, image_height = image.size

    # 创建画布对象
    canvas = Image.new('RGB', (image_width + 160, image_height), color='white')

    # 将原图绘制在画布的右侧
    canvas.paste(image, (200, 0))
    # 创建画笔和字体对象
    draw = ImageDraw.Draw(canvas)
    font_size = 50  # 调整字体大小
    font = ImageFont.truetype("arial.ttf", font_size)

    # 添加六行数字
    folder_names = ["B & E", 'F & PHI', 'Q & S', 'SE & X',"B & E", 'F & PHI', 'Q & S', 'SE & X']
    text_offset = 230

    for i, folder_name in enumerate(folder_names):
        text_position = (30, text_offset + i * 470)
        draw.text(text_position, folder_name, fill='black', font=font)

    # 保存或展示最终图片
    canvas.save("./figure_10/101.pdf")  # 如果想保存图片，将路径替换为输出路径
    canvas.show()
if __name__ == "__main__":
    main()
    draw_catalog()