from CycleGAN2 import *

if __name__ == '__main__':
    # Create a CycleGAN on GPU 0
    myCycleGAN = CycleGAN2(0)

    '''********* Model training *********'''
    trainA_dir = '/media/jared/TeamWork/ExperimentalData/WangShouju/FluoScans20200410/Patches_DAPI_SpGreen/ComplexTr'
    trainB_dir = '/media/jared/TeamWork/ExperimentalData/WangShouju/FluoScans20200410/Patches_DAPI_SpGreen/SpOrangeTr'
    models_dir = './models_DAPI_SpGreen_GAN'
    # output_sample_dir = './output/sample_MRI2DTI.png'
    batch_size = 20
    epochs = 10
    normalization_factor_A = 255
    normalization_factor_B = 255
    myCycleGAN.train(trainA_dir, normalization_factor_A, trainB_dir, normalization_factor_B, models_dir, batch_size,
                     epochs)

    '''********* Model testing *********'''
    # for epoch in range(10, 11, 1):
    #     G_X2Y_dir = '/media/jared/TeamWork/WangShouju/WSI-cycleGAN/models_DAPI_SpGreen_GAN/model_1_weights_epoch_' + str(epoch) + '.hdf5'
    #     print(G_X2Y_dir)
    #     test_X_dir = '/media/jared/TeamWork/ExperimentalData/WangShouju/FluoScans20200410/Patches_DAPI_SpGreen/ComplexTs/Test_00'
    #     synthetic_Y_dir = './outputs_DAPI_SpGreen_GAN/synthetic/synthetic_epoch_' + str(epoch)
    #     maybe_mkdir_p(synthetic_Y_dir)
    #     normalization_factor_X = 255
    #     normalization_factor_Y = 255
    #     myCycleGAN.synthesizeX2Y(G_X2Y_dir, test_X_dir, normalization_factor_X, synthetic_Y_dir, normalization_factor_Y)

