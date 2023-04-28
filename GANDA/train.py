from CycleGAN2_old import *

if __name__ == '__main__':
    # Create a CycleGAN on GPU 0
    myCycleGAN = CycleGAN2(0)

    for i in range(50, 51):


        trainA_dir = '/media/aiilab/Experiments/ExperimentalData/WangShouju/FluoScans20200410/Patches_DAPI_SpGreen/ComplexTr'
        trainB_dir = '/media/aiilab/Experiments/ExperimentalData/WangShouju/FluoScans20200410/Patches_DAPI_SpGreen/SpOrangeTr'
        models_dir = '/media/aiilab/Experiments/WangShouju/WSI-cycleGAN/cv/fluo5/models_GAN_%.2d' % i
        batch_size = 20
        epochs = 50
        normalization_factor_A = 255
        normalization_factor_B = 255

        myCycleGAN.train(trainA_dir, normalization_factor_A, trainB_dir, normalization_factor_B, models_dir, batch_size,
                         epochs)
