import numpy as np
import glob
import skimage.io
import PIL.Image
import PIL
import torch

def load_images2(path, shape):
    """
    Loads all the images in a file to as bw images to a single matrix concatenated in the 3rd dimension
    :param path: the path of the folder
    :return: matrix containing all the images in the folder
    """
    i = 1
    for im_path in glob.glob(path + "\\*.png"):
        if i == 1:
            images = skimage.io.imread(im_path, as_gray=True)
            if(images.shape != shape):
                images = np.array(PIL.Image.fromarray(images).resize(size=shape))
            images = images[:, :,np.newaxis]
            i = 2
        else:
            im = skimage.io.imread(im_path, as_gray=True)
            if(im.shape != shape):
                im = np.array(PIL.Image.fromarray(im).resize(size=shape))
            im = im[:, :, np.newaxis]
            images = np.concatenate((images, im), axis=2)
    return images


def load_images2(path, shape):
    """
    Loads all the images in a file to as bw images to a single matrix concatenated in the 3rd dimension
    :param path: the path of the folder
    :return: matrix containing all the images in the folder
    """
    i = 1
    for im_path in glob.glob(path + "\\*.png"):
        if i == 1:
            images = skimage.io.imread(im_path, as_gray=True)
            if(images.shape != shape):
                images = np.array(PIL.Image.fromarray(images).resize(size=shape))
            images = images[np.newaxis, np.newaxis, :, :]
            i = 2
        else:
            im = skimage.io.imread(im_path, as_gray=True)
            if(im.shape != shape):
                im = np.array(PIL.Image.fromarray(im).resize(size=shape))
            im = im[np.newaxis, np.newaxis, :, :]
            images = np.concatenate((images, im), axis=0)
    return images


def one_hot_labels(labels):
    ret = []
    for i in range(labels.size):
        if labels[i] == 1:
            ret.append([1, 0])
        elif labels[i] == 0:
            ret.append([0,1])
    ret = np.array(ret)
    return ret
