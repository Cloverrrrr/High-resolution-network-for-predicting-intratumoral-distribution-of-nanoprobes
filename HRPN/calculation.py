from skimage import io as io
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
# Image.MAX_IMAGE_PIXELS = None
from sklearn.metrics import r2_score, mean_absolute_error
import cv2
import math
import os

class calculation():

    def mse(img1, img2):
      mse = np.mean((img1 - img2) ** 2)
      return mse

    def mape(img1, img2):
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        mape = np.mean(np.abs((img2_array - img1_array) / img2_array))
        return  mape

    def psnr1(img1, img2):
        # compute mse
        mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
        # compute psnr
        if mse < 1e-10:
            return 100
        psnr1 = 20 * math.log10(255 / math.sqrt(mse))
        return psnr1


    def psnr2(img1, img2):
        mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
        if mse < 1e-10:
            return 100
        psnr2 = 20 * math.log10(1 / math.sqrt(mse))
        return psnr2




pic_dir = '/share/home/scz6051/Experiments/HRNet/yx3/data/pic_Ablation'
pic_list = os.listdir(pic_dir)
pic_list.sort()
print("All Pic_list:", pic_list)

true = io.imread("/share/home/scz6051/Experimental_Data/HRNet_data/TIF/Fluo5/SpOrange.tif")
i = 0
for pic in pic_list:
    print("Pic name:", pic)
    pic = io.imread(join(pic_dir, pic))
    (mean_pic, stddv_pic) = cv2.meanStdDev(pic)
    print("pic max value:", np.max(pic))
    print("pic min value:", np.min(pic))
    print("pic mean value:", np.mean(pic))
    print("pic cv2.mean value:", mean_pic)
    print("pic cv2.stddv value:", stddv_pic)

    mse = calculation.mse(true, pic)
    psnr1 = calculation.psnr1(true, pic)
    psnr2 = calculation.psnr2(true, pic)
    mae = mean_absolute_error(true, pic)
    r2_1 = r2_score(true, pic)
    r2_2 = r2_score(true, pic, multioutput='variance_weighted')


    print("PSNR1:", psnr1)
    print("PSNR2:", psnr2)
    print("MSE:", mse)
    print("RMSE:", np.sqrt(mse))
    print("MAE:", mae)
    print("R2_1:", r2_1)
    print("R2_2:", r2_2)
    i = i+1
    print("____________________________________________________________")

print("Finish")





