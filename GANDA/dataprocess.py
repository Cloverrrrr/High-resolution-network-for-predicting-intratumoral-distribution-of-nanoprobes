import os
import numpy as np
import utils
# import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
# from libtiff import TIFF
from PIL import Image
import scipy.misc as sic
import cv2
import matplotlib as plt
Image.MAX_IMAGE_PIXELS = None
## Image2Patchs
raw_dataset_dir = "/media/aiilab/Experiments/ExperimentalData/WangShouju/FluoScans20200410/TIF"
target_base = "/media/aiilab/Experiments/ExperimentalData/WangShouju/FluoScans20200410/Patches_B16_SpGreen"
target_ComplexTr = join(target_base, "ComplexTr")
target_SpOrangeTr = join(target_base, "SpOrangeTr")
# target_imagesVal = join(target_base, "imagesVal")
target_ComplexTs = join(target_base, "ComplexTs")
target_SpOrangeTs = join(target_base, "SpOrangeTs")
maybe_mkdir_p(target_ComplexTr)
maybe_mkdir_p(target_SpOrangeTr)
# maybe_mkdir_p(target_imagesVal)
maybe_mkdir_p(target_ComplexTs)
maybe_mkdir_p(target_SpOrangeTs)
caselist = ['Fluo_B16']
# caselist = os.listdir(raw_dataset_dir)
caselist.sort()
caselist.sort(key=lambda x: x[:-2])
print(caselist)
batch_size = 16
num_tr = 5
i = 0
print('image2patch...')
image2patch = utils.Image2Patch(patch_size=(512, 512), stride_size=(512, 512))
for casename in caselist:
    # DAPI = Image.open(join(raw_dataset_dir, casename, "DAPI.tif"))
    # DAPI_array = np.array(DAPI)
    # DAPI_array = DAPI_array[np.newaxis, np.newaxis, :, :]
    # DAPI_patches = image2patch.decompose(DAPI_array)

    SpGreen = Image.open(join(raw_dataset_dir, casename, "SpGreen.tif"))
    SpGreen_array = np.array(SpGreen)
    SpGreen_array = SpGreen_array[np.newaxis, np.newaxis, :, :]
    SpGreen_patches = image2patch.decompose(SpGreen_array)
    # print("max value:", np.max(DAPI_patches))
    # print("min value:", np.min(DAPI_patches))
    # print("mean value:", np.mean(DAPI_patches))

    # SpGreen_patches[SpGreen_patches < 50] = 0
    print("max value:", np.max(SpGreen_patches))
    print("min value:", np.min(SpGreen_patches))
    print("mean value:", np.mean(SpGreen_patches))

    # Complex_patches = DAPI_patches
    Complex_patches = SpGreen_patches
    # Complex_patches = np.concatenate((DAPI_patches, SpGreen_patches), axis=1)

    SpOrange = Image.open(join(raw_dataset_dir, casename, "SpOrange.tif"))
    SpOrange_array = np.array(SpOrange)
    SpOrange_array = SpOrange_array[np.newaxis, np.newaxis, :, :]
    SpOrange_patches = image2patch.decompose(SpOrange_array)
    # SpOrange_patches[SpOrange_patches < 50] = 0

    patch_avg = []
    for index in range(len(Complex_patches)):
        avg = np.mean(Complex_patches[index, :])
        patch_avg.append(avg)
    patch_avg = np.array(patch_avg)
    null_index = np.where(patch_avg <= 0)
    notnull_index = np.where(patch_avg > 0)

    np.savetxt(join(target_base, 'null_index_%02d.txt' % i), null_index[0])
    np.savetxt(join(target_base, 'notnull_index_%02d.txt' % i), notnull_index[0])

    Complex_patches = Complex_patches[notnull_index]
    SpOrange_patches = SpOrange_patches[notnull_index]
    Complex_patches = Complex_patches.astype('int16')
    SpOrange_patches = SpOrange_patches.astype('int16')
    print("max value:", np.max(SpOrange_patches))
    print("min value:", np.min(SpOrange_patches))
    print("mean value:", np.mean(SpOrange_patches))

    if i < num_tr:
        loops = len(SpOrange_patches) // batch_size
        reminder = len(SpOrange_patches) % batch_size
        for j in range(loops):
            Complex_name = join(target_ComplexTr, 'Complex_%02d_%03d.npy' % (i, j))
            SpOrange_name = join(target_SpOrangeTr, 'SpOrange_%02d_%03d.npy' % (i, j))
            raw_data = Complex_patches[j * batch_size: (j + 1) * batch_size]
            target_data = SpOrange_patches[j * batch_size: (j + 1) * batch_size]
            np.save(Complex_name, raw_data)
            np.save(SpOrange_name, target_data)
        if reminder != 0:
            Complex_name = join(target_ComplexTr, 'Complex_%02d_%03d.npy' % (i, loops))
            SpOrange_name = join(target_SpOrangeTr, 'SpOrange_%02d_%03d.npy' % (i, loops))
            raw_data = Complex_patches[loops * batch_size:]
            target_data = SpOrange_patches[loops * batch_size:]
            np.save(Complex_name, raw_data)
            np.save(SpOrange_name, target_data)
    else:
        loops = len(SpOrange_patches) // batch_size
        reminder = len(SpOrange_patches) % batch_size
        for j in range(loops):
            Complex_name = join(target_ComplexTs, 'Test_%02d' % (i-num_tr), 'Complex_%02d_%03d.npy' % (i-num_tr, j))
            SpOrange_name = join(target_SpOrangeTs, 'Test_%02d' % (i-num_tr), 'SpOrange_%02d_%03d.npy' % (i-num_tr, j))
            raw_data = Complex_patches[j * batch_size: (j + 1) * batch_size]
            target_data = SpOrange_patches[j * batch_size: (j + 1) * batch_size]
            np.save(Complex_name, raw_data)
            np.save(SpOrange_name, target_data)
        if reminder != 0:
            Complex_name = join(target_ComplexTs, 'Test_%02d' % (i-num_tr), 'Complex_%02d_%03d.npy' % (i-num_tr, loops))
            SpOrange_name = join(target_SpOrangeTs, 'Test_%02d' % (i-num_tr), 'SpOrange_%02d_%03d.npy' % (i-num_tr, loops))
            raw_data = Complex_patches[loops * batch_size:]
            target_data = SpOrange_patches[loops * batch_size:]
            np.save(Complex_name, raw_data)
            np.save(SpOrange_name, target_data)
    i += 1

print('Image2Patch转化完成')




# # #======Patches2Image=====
test_data_dir = '/media/aiilab/Experiments/ExperimentalData/WangShouju/FluoScans20200410/Patches_B16_SpGreen'
syn_data_dir = '/media/aiilab/Experiments/WangShouju/WSI-cycleGAN/outputs_B16_SpGreen_GAN/synthetic'
caseList = os.listdir(join(test_data_dir, 'SpOrangeTs'))
Image.MAX_IMAGE_PIXELS = None
print(caseList)
print('patches2image...')
image2patch = utils.Image2Patch(patch_size=(512, 512), stride_size=(512, 512))
SpOrange_dir = '/media/aiilab/Experiments/ExperimentalData/WangShouju/FluoScans20200410/TIF/Fluo_B16'
synSpOrange_dir = '/media/aiilab/Experiments/WangShouju/WSI-cycleGAN/outputs_B16_SpGreen_GAN/synthetic/synthetic_epoch_50'
# synSpOrange_dir = '/media/jared/Experiments/Experimental Data/FluoScans20200410/Patches/SpOrangeTs/Test_00'


for casename in caseList:
    notnull_index = np.loadtxt(join(test_data_dir, 'notnull_index_00.txt'))
    notnull_index = notnull_index.astype(int)
    patchList = os.listdir(synSpOrange_dir)
    patchList.sort()
    patchList.sort(key=lambda x: x[:-4])

    SpOrange = Image.open(join(SpOrange_dir, "SpOrange.tif"))
    SpOrange_array = np.array(SpOrange)
    SpOrange_array = SpOrange_array[np.newaxis, np.newaxis, :, :]
    SpOrange_patches = image2patch.decompose(SpOrange_array)
    # SpOrange_patches = np.load(join(SpOrange_dir, 'SpOrange.npy'))
    synSpOrange = np.zeros(SpOrange_patches.shape, dtype=np.float32)
    i = 0
    for patchname in patchList:
        patch_array = np.load(join(synSpOrange_dir, patchname))
        patch_array = np.transpose(patch_array, [0, 3, 1, 2])
        if i == 0:
            synSpOrange_part = patch_array
        else:
            synSpOrange_part = np.append(synSpOrange_part, patch_array, axis=0)
        i = i+1
    synSpOrange[notnull_index] = synSpOrange_part
    synSpOrange = image2patch.compose(synSpOrange)
    synSpOrange = np.squeeze(synSpOrange)
    print("max value:", np.max(synSpOrange))
    print("min value:", np.min(synSpOrange))
    print("mean value:", np.mean(synSpOrange))
    synSpOrange_img = Image.fromarray(synSpOrange.astype(np.uint8))
    print("max value:", np.max(synSpOrange_img))
    print("min value:", np.min(synSpOrange_img))
    print("mean value:", np.mean(synSpOrange_img))

    synSpOrange_img.save(join(synSpOrange_dir, 'synSpOrange.tif'))
#     # Image.save(synSpOrange_dir, synSpOrange_img)
#     # np.save(SpOrange_name, target_data)
