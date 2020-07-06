import sys
sys.path.append('../FDFAPBG')
import math
import matplotlib.pyplot as plt
from numpy import random

def plot_images(images, img_name, titles=None, cols=3, save_plt=False, file_format="png", dpi=None):
    rows = math.ceil(len(images) / cols)
    for i in range(len(images)):
        index = i + 1
        plt.subplot(rows, cols, index)
        plt.imshow(images[i])
        if titles is not None:
            plt.title(img_name + ": " + str(titles[i]))
        # plt.xticks([])
        # plt.yticks([])
    if save_plt:
        plt.savefig(
            "outputs/images/{}".format(img_name), dpi=dpi, format=file_format)
    else:
        plt.show()

def plot_image(image, img_name, figsize=(5,5), save_plt=False, file_format="png", dpi=None):
    """
        figsize: (5,5) 5cm x 5cm
        dpi: i.e 300 
    """
    plt.figure(figsize=figsize)
    img = plt.imshow(image)
    # img.set_cmap('hot')
    plt.axis('off')
    if save_plt:
        plt.savefig("outputs/images/{}".format(img_name), bbox_inches='tight', pad_inches=0)
    else:
        plt.show()