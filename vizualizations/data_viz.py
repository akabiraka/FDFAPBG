import sys
sys.path.append('../FDFAPBG')
import math
import matplotlib.pyplot as plt

def plot_images(images, img_name, titles=None, cols=3, save_plt=False):
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
            "../output_images/contact_maps_predicted_real/{}.png".format(img_name))
    else:
        plt.show()