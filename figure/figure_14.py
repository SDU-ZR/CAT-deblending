import os

image_dir = './figure_14'

# 大图的尺寸
rows, cols = 16, 6
# 获取图片列表
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# 定义大图尺寸和子图间隙
fig, axes = plt.subplots(rows, cols, figsize=(40, 55))
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0, hspace=0)
# 读取并放置96张图像
for i in range(min(rows * cols, len(image_files))):
    img_path = os.path.join(image_dir, image_files[i])
    img = mpimg.imread(img_path)
    row = i // cols
    col = i % cols
    axes[row, col].imshow(img)
    axes[row, col].axis('off')
plt.show()
plt.savefig("./14.png", dpi=100)