"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import copy
import torch
# import openslide
import cv2
from PIL import ImageFilter, Image
try:
    _imread = scipy.misc.imread
except AttributeError:
    from imageio import imread as _imread

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import nibabel as nib
from keras.layers import Dropout, Layer, Input, Conv2D, Activation, add, BatchNormalization, Conv2DTranspose, UpSampling2D
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# from keras_contrib.layers.normalization.instancenormalization import InputSpec
from keras.engine.topology import InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.backend import mean
from keras.models import Model
from keras.engine.topology import Network
import imageio
import numpy as np
import random
import datetime
import time
import math
import sys
import keras.backend as K
import tensorflow as tf
import datetime
import nibabel as nib

# import tensorflow as tf
import matplotlib.pyplot as plt


def discriminator_loss(loss_object, real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(loss_object, fake_output):
    return loss_object(tf.ones_like(fake_output), fake_output)


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def resize(image, height, width):
    image = tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image


def convert_onehot(label, num_classes):
    return tf.one_hot(label, num_classes)


def preprocess_image(data, img_shape, num_classes):
    image = data['image']
    image = resize(image, img_shape[0], img_shape[1])
    image = normalize(image)

    label = convert_onehot(data['label'], num_classes)

    return image, label



def normalize_data(data, normalization_factor):
    # Normalize data to [-1, 1]
    print("max value:", np.max(data))
    if np.array(normalization_factor).size == 1:
        data = data/normalization_factor
    else:
        for i in range(data.shape[2]):
            data[:, :, i, :] = data[:, :, i, :]/normalization_factor[i] # normalize data for each channel
    data = data*2-1
    return data

def denormalize_data(data, normalization_factor):
    # Denormalize data to [-1, 1]
    data = (data+1)/2
    if np.array(normalization_factor).size == 1:
        data = data*normalization_factor
    else:
        for i in range(data.shape[2]):
            data[:, :, i, :] = data[:, :, i, :]*normalization_factor[i] # normalize data for each channel
    return data

def load_data(data_dir, normalization_factor):
        data = nib.load(data_dir).get_fdata()
        data = np.pad(data,((13,13),(13,13),(0,0)),'constant')
        data = np.expand_dims(data, axis=0)
        # data = np.load(data_dir,allow_pickle=True)
        data[data < 0] = 0
        if data.ndim == 2:
            data = data[:, :, np.newaxis, np.newaxis]
        data = normalize_data(data, normalization_factor)
        # data = np.transpose(data, (0, 2, 3, 1))
        data = np.transpose(data, (3, 1, 2, 0))
        print('Loading data, data size: {}, number of data: {}'.format(data.shape[1:4], data.shape[0]))
        print("max value:", np.max(data))
        print("min value:", np.min(data))
        # Make sure that slice size is multiple 4
        # if (data.shape[1]%4 != 0):
        #     data = np.append(data, np.zeros((data.shape[0], 4-data.shape[1]%4, data.shape[2], data.shape[3]))-1, axis=1)
        # if (data.shape[3] % 4 != 0):
        #     data = np.append(data,np.zeros((data.shape[0], data.shape[1] , data.shape[2],4-data.shape[3]%4))-1,axis=1)
        #
        # if (data.shape[2]%4 != 0):
        #     data = np.append(data, np.zeros((data.shape[0], data.shape[1], 4-data.shape[2]%4, data.shape[3]))-1, axis=2)

        return data

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)
    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])
    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))
            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))
        return return_images

class Image2Patch(object):
    def __init__(self, patch_size, stride_size):
        self.patch_size = patch_size
        self.stride_size = stride_size

    def _paint_border_overlap(self, image):
        patch_h, patch_w = self.patch_size
        stride_h, stride_w = self.stride_size
        img_h, img_w = image.shape[2], image.shape[3]

        left_h = (img_h - patch_h) % stride_h
        left_w = (img_w - patch_w) % stride_w

        pad_h = stride_h - left_h if left_h > 0 else 0
        pad_w = stride_w - left_w if left_w > 0 else 0

        # padding = torch.nn.ZeroPad2d((0, pad_w, 0, pad_h))
        # image = padding(image)
        image = np.pad(image, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), 'constant', constant_values=(0, 0))
        self.new_size = (image.shape[2], image.shape[3])
        return image

    def _extract_order_overlap(self, image):
        patch_h, patch_w = self.patch_size
        stride_h, stride_w = self.stride_size
        N_img, ch, img_h, img_w = image.shape

        assert ((img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0)
        N_patch_imgs = ((img_h - patch_h) // stride_h + 1) * ((img_w - patch_w) // stride_w + 1)
        N_patch_total = N_patch_imgs*N_img

        patches = np.zeros((N_patch_total, ch, patch_h, patch_w), dtype=float)
        count = 0
        for i in range(N_img):
            for h in range((img_h - patch_h) // stride_h + 1):
                for w in range((img_w - patch_w) // stride_w + 1):
                    patch = image[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w]
                    patches[count] = patch
                    count += 1
        assert (count == N_patch_total)
        return patches

    def decompose(self, image):
        """
        Args:
            image: <B_i x C x H_i x W_i>
        Returns:
            patches: <B_p x C x H_p x W_p>
        """
        assert (len(image.shape) == 4)
        self.image_size = (image.shape[2], image.shape[3])
        image = self._paint_border_overlap(image)
        patches = self._extract_order_overlap(image)

        self.decomposed = True
        return patches

    def compose(self, patches):
        assert (self.decomposed == True)
        ch, patch_h, patch_w = patches.shape[1], patches.shape[2], patches.shape[3]
        img_h, img_w = self.new_size
        stride_h, stride_w = self.stride_size

        N_patches_h = (img_h - patch_h) // stride_h + 1
        N_patches_w = (img_w - patch_w) // stride_w + 1
        N_patches_img = N_patches_h * N_patches_w

        assert (patches.shape[0] % N_patches_img == 0)
        N_img = patches.shape[0] // N_patches_img

        full_prob = torch.zeros((N_img, ch, img_h, img_w))
        # full_sum = torch.zeros((N_img, ch, img_h, img_w))

        count = 0
        for i in range(N_img):
            for h in range((img_h - patch_h) // stride_h + 1):
                for w in range((img_w - patch_w) // stride_w + 1):
                    full_prob[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += \
                    patches[count]
                    # full_sum[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += 1
                    count += 1

        assert (count == patches.shape[0])
        image = full_prob
        image = image[:, :, :self.image_size[0], :self.image_size[1]]
        self.decomposed = False
        image = image.numpy()
        return image
