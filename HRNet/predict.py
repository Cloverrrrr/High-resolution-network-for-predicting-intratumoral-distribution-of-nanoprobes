import numpy as np
import torch
import os
import sys
sys.path.append('/share/home/scz6051/Experiments/HRNet/yx3/utils')
from utils.configer import Configer
from model import hrnet
from batchgenerators.utilities.file_and_folder_operations import *

if __name__ == "__main__":
    configer = Configer(configs='utils/H_48_D_4.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = hrnet.HRNet_W48(configer)
    net.to(device=device)
    net.load_state_dict(torch.load('/share/home/scz6051/Experiments/HRNet/yx3/Ablation_stage2_2.pth', map_location=device))
    net.eval()
    raw_dir = '/share/home/scz6051/Experimental_Data/HRNet_data/Patches_DAPI_SpGreen_16'
    tests_dir = os.path.join(raw_dir, 'ComplexVal')
    tests_path = os.listdir(tests_dir)
    tests_path.sort()
    tests_path.sort(key=lambda x: x[:-4])
    print(tests_path)
    for test_path in tests_path:
        file_name = test_path.split('.')[0] + '_syn.npy'
        img = np.load(os.path.join(tests_dir, test_path), allow_pickle=True)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)

        pred = net(img_tensor)
        predict_dir = '/share/home/scz6051/Experiments/HRNet/yx3/data/Ablation_stage2_2/outputs_DAPI_SpGreen/synthetic_16'
        maybe_mkdir_p(predict_dir)
        np.save(os.path.join(predict_dir, file_name), pred.data.cpu())
        print("finish")




