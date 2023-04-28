import os
import glob
from torch.utils.data import Dataset


import numpy as np


class Unet_Loader(Dataset):
    def __init__(self, data_path, training=True):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.npy'))
        self.training = training


    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        if self.training == True:
            label_path = image_path.replace('ComplexTr', 'SpOrangeTr')
            label_path = label_path.replace('Complex_', 'SpOrange_')
        if self.training == False:
            label_path = image_path.replace('ComplexVal', 'SpOrangeVal')
            label_path = label_path.replace('Complex_', 'SpOrange_')

        image = np.load(image_path,allow_pickle=True).squeeze(0)
        label = np.load(label_path, allow_pickle=True).squeeze(0)

        return image, label

    def __len__(self):
        return len(self.imgs_path)




