# coding=utf-8
# 导入相应的python包
import argparse
from skimage import io
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries

image = img_as_float(io.imread('D:\python\photo/blq1.jpg'))

segments = slic(image, n_segments=800, sigma=5)
fig = plt.figure("Superpixels -- %d segments" % (400))
plt.subplot(131)
plt.title('image')
plt.imshow(image)
# plt.subplot(132)
# plt.title('segments')
# plt.imshow(segments)
plt.subplot(133)
plt.title('image and segments')
plt.imshow(mark_boundaries(image, segments))
plt.show()

