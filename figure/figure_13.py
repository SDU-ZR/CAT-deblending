"""
为了展示成功解混但是错误的三种模式
"""
from PIL import Image
import matplotlib.pyplot as plt
import os

# List of image filenames
image_filenames = sorted(os.listdir("./figure_13/sucess_fale_desi_dr9"))
print(image_filenames)

# Load the images into a list
image_list = [Image.open(os.path.join("./figure_13/sucess_fale_desi_dr9", filename)) for filename in image_filenames]

# Check the size of the first image as a reference
ref_width, ref_height = image_list[0].size

# Plotting the images in a 3x3 grid
fig, axes = plt.subplots(4, 3, figsize=(12, 8.8

                                        ))

for i, ax in enumerate(axes.flatten()):
    print(i)
    ax.imshow(image_list[i])
    ax.axis('off')
    # ax.set_title(f"Image {i+1}")

plt.tight_layout()
plt.savefig("./13.pdf", dpi=100)
plt.show()