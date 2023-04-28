import numpy as np
import dataprocess_utils as utils
from batchgenerators.utilities.file_and_folder_operations import *
from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = None



#
## Image2Patchs
raw_dataset_dir = "/share/home/scz6051/Experimental_Data/HRNet_data/TIF"
target_base = "/share/home/scz6051/Experimental_Data/HRNet_data/Patches_DAPI_SpGreen"
maybe_mkdir_p(target_base)
target_ComplexTr = join(target_base, "ComplexTr")
target_SpOrangeTr = join(target_base, "SpOrangeTr")
target_ComplexVal = join(target_base, "ComplexVal")
target_SpOrangeVal = join(target_base, "SpOrangeVal")
target_ComplexTs = join(target_base, "ComplexTs")
target_SpOrangeTs = join(target_base, "SpOrangeTs")
maybe_mkdir_p(target_ComplexTr)
maybe_mkdir_p(target_SpOrangeTr)
maybe_mkdir_p(target_ComplexVal)
maybe_mkdir_p(target_SpOrangeVal)
maybe_mkdir_p(target_ComplexTs)
maybe_mkdir_p(target_SpOrangeTs)
# caselist = ['Fluo5']
caselist = os.listdir(raw_dataset_dir)
caselist.sort()
print(caselist)
batch_size = 1
num_val = 5
i = 0
print('image2patch...')
image2patch = utils.Image2Patch(patch_size=(512, 512), stride_size=(512, 512))
for casename in caselist:
    DAPI = Image.open(join(raw_dataset_dir, casename, "DAPI.tif"))
    DAPI_array = np.array(DAPI)
    DAPI_array = DAPI_array[np.newaxis, np.newaxis, :, :]
    DAPI_patches = image2patch.decompose(DAPI_array)
    print("max value:", np.max(DAPI_patches))
    print("min value:", np.min(DAPI_patches))
    print("mean value:", np.mean(DAPI_patches))

    SpGreen = Image.open(join(raw_dataset_dir, casename, "SpGreen.tif"))
    SpGreen_array = np.array(SpGreen)
    SpGreen_array = SpGreen_array[np.newaxis, np.newaxis, :, :]
    SpGreen_patches = image2patch.decompose(SpGreen_array)
    print("max value:", np.max(SpGreen_patches))
    print("min value:", np.min(SpGreen_patches))
    print("mean value:", np.mean(SpGreen_patches))

    SpOrange = Image.open(join(raw_dataset_dir, casename, "SpOrange.tif"))
    SpOrange_array = np.array(SpOrange)
    SpOrange_array = SpOrange_array[np.newaxis, np.newaxis, :, :]
    SpOrange_patches = image2patch.decompose(SpOrange_array)
    print("max value:", np.max(SpOrange_patches))
    print("min value:", np.min(SpOrange_patches))
    print("mean value:", np.mean(SpOrange_patches))

    # Complex_patches = DAPI_patches
    # Complex_patches = SpGreen_patches
    Complex_patches = np.concatenate((DAPI_patches, SpGreen_patches), axis=1)


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


    if i == num_val:
        loops = len(SpOrange_patches) // batch_size
        reminder = len(SpOrange_patches) % batch_size
        for j in range(loops):
            Complex_name = join(target_ComplexVal, 'Complex_%02d_%04d.npy' % (i, j))
            SpOrange_name = join(target_SpOrangeVal, 'SpOrange_%02d_%04d.npy' % (i, j))
            raw_data = Complex_patches[j * batch_size: (j + 1) * batch_size]
            target_data = SpOrange_patches[j * batch_size: (j + 1) * batch_size]
            np.save(Complex_name, raw_data)
            np.save(SpOrange_name, target_data)
        if reminder != 0:
            Complex_name = join(target_ComplexVal, 'Complex_%02d_%04d.npy' % (i, loops))
            SpOrange_name = join(target_SpOrangeVal, 'SpOrange_%02d_%04d.npy' % (i, loops))
            raw_data = Complex_patches[loops * batch_size:]
            target_data = SpOrange_patches[loops * batch_size:]
            np.save(Complex_name, raw_data)
            np.save(SpOrange_name, target_data)
    if i == 5:
        loops = len(SpGreen_patches) // batch_size
        reminder = len(SpGreen_patches) % batch_size
        for j in range(loops):
            Complex_name = join(target_ComplexTs, 'Complex_%02d_%04d.npy' % (i, j))
            SpOrange_name = join(target_SpOrangeTs, 'SpOrange_%02d_%04d.npy' % (i, j))
            raw_data = Complex_patches[j * batch_size: (j + 1) * batch_size]
            target_data = SpOrange_patches[j * batch_size: (j + 1) * batch_size]
            np.save(Complex_name, raw_data)
            np.save(SpOrange_name, target_data)
        if reminder != 0:
            Complex_name = join(target_ComplexTs, 'Complex_%02d_%04d.npy' % (i, loops))
            SpOrange_name = join(target_SpOrangeTs, 'SpOrange_%02d_%04d.npy' % (i, loops))
            raw_data = Complex_patches[loops * batch_size:]
            target_data = SpOrange_patches[loops * batch_size:]
            np.save(Complex_name, raw_data)
            np.save(SpOrange_name, target_data)
    else:
        loops = len(SpOrange_patches) // batch_size
        reminder = len(SpOrange_patches) % batch_size
        for j in range(loops):
            Complex_name = join(target_ComplexTr, 'Complex_%02d_%04d.npy' % (i, j))
            SpOrange_name = join(target_SpOrangeTr, 'SpOrange_%02d_%04d.npy' % (i, j))
            raw_data = Complex_patches[j * batch_size: (j + 1) * batch_size]
            target_data = SpOrange_patches[j * batch_size: (j + 1) * batch_size]
            np.save(Complex_name, raw_data)
            np.save(SpOrange_name, target_data)
        if reminder != 0:
            Complex_name = join(target_ComplexTr, 'Complex_%02d_%04d.npy' % (i, loops))
            SpOrange_name = join(target_SpOrangeTr, 'SpOrange_%02d_%04d.npy' % (i, loops))
            raw_data = Complex_patches[loops * batch_size:]
            target_data = SpOrange_patches[loops * batch_size:]
            np.save(Complex_name, raw_data)
            np.save(SpOrange_name, target_data)


    i += 1

print('Image2Patch转化完成')




# #======Patches2Image=====
test_data_dir = '/home/cloudam/Experimental_Data/HRNet_data_cv_fluo0_16/Patches_DAPI_SpGreen'
syn_data_dir = '/home/cloudam/Experiments/HRNet/yx3/data/fluo0/outputs_DAPI_SpGreen'
caseList = os.listdir(os.path.join(test_data_dir, 'SpOrangeVal'))
caseList.sort()
# print(caseList)
Image.MAX_IMAGE_PIXELS = None
f = None
print('patches2image...')
image2patch = utils.Image2Patch(patch_size=(512, 512), stride_size=(512, 512))
SpOrange_dir = '/share/home/scz6051/Experimental_Data/HRNet_data/TIF/Fluo0'
synSpOrange_dir ='/share/home/scz6051/Experiments/HRNet/yx3/data/fluo0/outputs_DAPI_SpGreen/synthetic_val_16'


notnull_index = np.loadtxt(os.path.join(test_data_dir, 'notnull_index_00.txt'))
notnull_index = notnull_index.astype(int)
patchList = os.listdir(synSpOrange_dir)
patchList.sort()
print(patchList)

SpOrange = Image.open(os.path.join(SpOrange_dir, "SpOrange.tif"))
SpOrange_array = np.array(SpOrange)
SpOrange_array = SpOrange_array[np.newaxis, np.newaxis, :, :]
SpOrange_patches = image2patch.decompose(SpOrange_array)
synSpOrange = np.zeros(SpOrange_patches.shape, dtype=np.float32)
i = 0
for patchname in patchList:
    patch_array = np.load(join(synSpOrange_dir, patchname), allow_pickle=True)
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

synSpOrange_img.save(os.path.join(syn_data_dir, 'val_synSpOrange.tif'))
print("finish")




# #======Patches2Image=====
test_data_dir = '/share/home/scz6051/Experimental_Data/HRNet_data/Patches_DAPI_SpGreen_16'
# syn_data_dir = '/share/home/scz6051/Experiments/HRNet/yx3/data/fluo0/outputs_DAPI_SpGreen'
# caseList = os.listdir(os.path.join(test_data_dir, 'SpOrangeVal'))
# caseList = os.listdir(test_data_dir)
Image.MAX_IMAGE_PIXELS = None
# print(caseList)
print('patches2image...')
image2patch = utils.Image2Patch(patch_size=(512, 512), stride_size=(512, 512))
SpOrange_dir = '/share/home/scz6051/Experimental_Data/HRNet_data/TIF/Fluo5'  #val是多少就改多少
synSpOrange_dir ='/share/home/scz6051/Experiments/HRNet/yx3/data/Ablation_stage1/outputs_DAPI_SpGreen/synthetic_16'
n = 0
for n in range(1):
    notnull_index = np.loadtxt(join(test_data_dir, 'notnull_index_05.txt'))  #val是多少就改多少
    notnull_index = notnull_index.astype(int)
    patchList = os.listdir(synSpOrange_dir)
    patchList.sort()
    patchList.sort(key=lambda x: x[:-4])

    SpOrange = Image.open(join(SpOrange_dir, "SpOrange.tif"))
    SpOrange_array = np.array(SpOrange)
    SpOrange_array = SpOrange_array[np.newaxis, np.newaxis, :, :]
    SpOrange_patches = image2patch.decompose(SpOrange_array)
    synSpOrange = np.zeros(SpOrange_patches.shape, dtype=np.float32)
    i = 0
    for patchname in patchList:
        patch_array = np.load(join(synSpOrange_dir, patchname))
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

    synSpOrange_img.save(join("/share/home/scz6051/Experiments/HRNet/yx3/data/pic_Ablation", 'Ablation_stage1_DAPI_SpGreen.tif'))

