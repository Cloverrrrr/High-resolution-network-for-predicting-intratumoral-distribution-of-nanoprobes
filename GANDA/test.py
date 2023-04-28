from CycleGAN2_old import *

if __name__ == '__main__':
    # Create a CycleGAN on GPU 0
    myCycleGAN = CycleGAN2(-1)


    for epoch in range(50, 51, 1):
        G_X2Y_dir = '/media/aiilab/Experiments/WangShouju/WSI-cycleGAN/cv/fluo5/models_GAN_50/G_A2B_weights_epoch_' + str(epoch) + '.hdf5'
        print(G_X2Y_dir)
        test_X_dir = '/media/aiilab/Experiments/ExperimentalData/WangShouju/FluoScans20200410/Patches_DAPI_SpGreen/ComplexTs/Test_00'
        test_Y_dir = '/media/aiilab/Experiments/ExperimentalData/WangShouju/FluoScans20200410/Patches_DAPI_SpGreen/SpOrangeTs/Test_00'
        synthetic_Y_dir = '/media/aiilab/Experiments/WangShouju/WSI-cycleGAN/cv/fluo5/test'
        maybe_mkdir_p(synthetic_Y_dir)
        normalization_factor_X = 255
        normalization_factor_Y = 255
        valloss = myCycleGAN.synthesizeX2Y(G_X2Y_dir, test_X_dir, test_Y_dir, normalization_factor_X, synthetic_Y_dir, normalization_factor_Y)
        valloss.append(valloss)
        print('Epoch:', epoch)