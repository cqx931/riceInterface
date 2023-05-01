import matplotlib.pyplot as plt
import glob
import os
import random

def getRandomFile(path):
    # Get List of all images
    files = glob.glob(path + '/**/*.jpg', recursive=True)
    return random.choice(files)


## Function to plot multiple images
def plot_img(images, titles):
    fig, axs = plt.subplots(nrows = 1, ncols = len(images), figsize = (15, 15))
    for i, p in enumerate(images):
        axs[i].imshow(p, 'gray')
        axs[i].set_title(titles[i])
        #axs[i].axis('off')
    plt.show()