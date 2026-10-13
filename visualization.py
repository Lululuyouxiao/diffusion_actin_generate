import cv2
import numpy as np
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
from skimage.measure import label
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from matplotlib.pyplot import MultipleLocator
from skimage.measure import label, regionprops
import math
from scipy.stats import norm
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit


def visualization(average_num, colormap):
    font = {'family': 'Arial',
            'size': 12,
            'color': 'white',
            }
    average_num = average_num
    test_num = 100
    save_dir = './results/visualization/'
    for idImage in range(test_num):
        platte = np.zeros([256, 256])
        for idGenerate in range(average_num):
            sf_dir = 'results/' + 'example/' + '%d/' % idGenerate
            predict_img = cv2.imread(sf_dir + '%04d.png' % idImage)
            predict_img = cv2.cvtColor(predict_img, cv2.COLOR_BGR2GRAY)
            predict_img = predict_img > threshold_otsu(predict_img)
            predict_img = predict_img * 255
            platte += predict_img
        platte = platte // average_num
        platte = np.array(platte, dtype=np.uint8)
        platte[0][0] = 255
        platte = platte / 255
        fig, ax = plt.subplots()
        axins = inset_axes(ax,
                           width="40%",  # width = 50% of parent_bbox width
                           height="3%",  # height : 5%
                           loc=4,
                           borderpad=1.8
                           )
        im = ax.imshow(platte, cmap=mpl.colormaps[colormap])
        # fig.colorbar(im, cax=axins, orientation="horizontal", ticks=[0, 0.5, 1])
        cb = plt.colorbar(im, cax=axins, orientation="horizontal", ticks=[0.0, 0.5, 1.0])
        cb.ax.set_title('P', fontdict=font)
        cb.ax.tick_params(colors='white')
        axins.xaxis.set_ticks_position("bottom")
        # plt.axis('off')
        ax.axis('off')
        # plt.show()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + '%04d.png' % idImage, dpi=300, bbox_inches='tight', pad_inches=-0.01)
        # cv2.imshow('test', platte)
        # cv2.waitKey(0)
        # print(probability)
        plt.cla()
        plt.clf()
        plt.close()

