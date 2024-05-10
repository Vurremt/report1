import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Open and return both images
def open_2_images(path, img_1, img_2):
    img1 = Image.open(path + img_1)
    img2 = Image.open(path + img_2)
    return img1, img2

# With two images and their points, draw corresponding lines between both
def concate_img_1_2(img1, img2, points1, points2, colors):
    fig_concate = plt.figure(figsize=(10, 5))

    # Concate both images for linked lines
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    img = np.hstack((img1_np, img2_np))
    ax = fig_concate.add_subplot(1, 1, 1)
    ax.imshow(img)

    # Draw connecting lines
    for i in range(len(points1)):
        ax.plot([points1[i][0], points2[i][0] + img1.size[0]], [points1[i][1], points2[i][1]], color=colors[i])

# Show the buffer image and save it in a folder
def save_and_show(path, final_img):
    plt.savefig(path + final_img)
    plt.show()
