import os
from typing import Optional, Any

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import nibabel as nib
from keras.layers import Dropout, Layer, Input, Conv2D, Activation, add, BatchNormalization, Conv2DTranspose, UpSampling2D, MaxPooling2D, concatenate
from keras.layers.core import Dense, Flatten
import keras
from keras.utils.np_utils import to_categorical
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# from keras_contrib.layers.normalization.instancenormalization import InputSpec
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
import SimpleITK as sitk

import datetime
from utils import *
from batchgenerators.utilities.file_and_folder_operations import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0, 1'

print('CycleGAN loaded...')

class CycleGAN2():
    def __init__(self, selected_gpu):
        os.environ["CUDA_VISIBLE_DEVICES"]=str(selected_gpu)
        print('Initializing a CycleGAN on GPU ' + os.environ["CUDA_VISIBLE_DEVICES"])
        # self.normalization = InstanceNormalization
        # Hyper parameters
        self.lr_D = 2e-4
        self.lr_G = 2e-4
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.lambda_1 = 1.0  # Cyclic loss weight A_2_B
        self.lambda_2 = 10.0  # Cyclic loss weight B_2_A
        self.lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images
        self.supervised_weight = 10
        self.synthetic_pool_size = 50
        # optimizer
        self.opt_D = Adam(self.lr_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.lr_G, self.beta_1, self.beta_2)

        # TensorFlow wizardry
        # config = tf.ConfigProto()
        # Don't pre-allocate memory; allocate as-needed
        # config.gpu_options.allow_growth = True
        # Create a session with the above options specified.
        # session = tf.Session(config=config)
        # session = tf.Session()
        # K.tensorflow_backend.set_session(session)

    def create_discriminator_and_generator(self):
        print('Creating Discriminator and Generator ...')
        # Discriminator
        # D_A = self.Discriminator()
        D_B = self.Discriminator()
        loss_weights_D = [0.5]
        # image_A = Input(shape=self.data_shapeA)
        image_B = Input(shape=self.data_shapeB)
        # guess_A = D_A(image_A)
        guess_B = D_B(image_B)
        # self.D_A = Model(inputs=image_A, outputs=guess_A, name='D_A')
        self.D_B = Model(inputs=image_B, outputs=guess_B, name='D_B')
        # self.D_A.compile(optimizer=self.opt_D, loss=self.lse, loss_weights=loss_weights_D)
        self.D_B.compile(optimizer=self.opt_D, loss=self.lse, loss_weights=loss_weights_D)
        # Use containers to avoid falsy keras error about weight descripancies
        # self.D_A_static = Network(inputs=image_A, outputs=guess_A, name='D_A_static')
        self.D_B_static = Network(inputs=image_B, outputs=guess_B, name='D_B_static')
        # Do note update discriminator weights during generator training
        # self.D_A_static.trainable = False
        self.D_B_static.trainable = False

        # Generators
        self.G_A2B = self.Generator(name='G_A2B')
        # self.G_B2A = self.Generator(name='G_B2A')
        real_A = Input(shape=self.data_shapeA, name='real_A')
        # real_B = Input(shape=self.data_shapeB, name='real_B')
        synthetic_B = self.G_A2B(real_A)
        # synthetic_A = self.G_B2A(real_B)
        # dA_guess_synthetic = self.D_A_static(synthetic_A)
        dB_guess_synthetic = self.D_B_static(synthetic_B)
        # reconstructed_A = self.G_B2A(synthetic_B)
        # reconstructed_B = self.G_A2B(synthetic_A)
        # model_outputs = [synthetic_B]
        model_outputs = [dB_guess_synthetic]
        compile_losses = [self.lse]
        compile_weights = [self.lambda_1]
        # compile_weights = [self.lambda_D]
        # model_outputs.append(dA_guess_synthetic)
        # model_outputs.append(dB_guess_synthetic)

        if self.use_supervised_learning:
            # model_outputs.append(synthetic_A)
            model_outputs.append(synthetic_B)
            compile_losses.append('MSE')
            # compile_losses.append('MAE')
            compile_weights.append(self.supervised_weight)
            # compile_weights.append(self.supervised_weight)
        self.G_model = Model(inputs=[real_A], outputs=model_outputs, name='G_model')
        self.G_model.compile(optimizer=self.opt_G, loss=compile_losses, loss_weights=compile_weights)


    def ck(self, x, k, use_normalization, stride):
            x = Conv2D(filters=k, kernel_size=4, strides=stride, padding='same', use_bias=True)(x)
            # Normalization is not done on the first discriminator layer
            if use_normalization:
                x = BatchNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
            x = LeakyReLU(alpha=0.2)(x)
            return x

    def c7Ak(self, x, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid', use_bias=True)(x)
        x = BatchNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def dk(self, x, k):
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same', use_bias=True)(x)
        x = BatchNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def Rk(self, x0):
        k = int(x0.shape[-1])
        # first layer
        x = ReflectionPadding2D((1,1))(x0)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=True)(x)
        x = BatchNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=True)(x)
        x = BatchNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = add([x, x0])
        return x

    def uk(self, x, k):
        if self.use_resize_convolution:
            x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            # x = ReflectionPadding2D((1, 1))(x)
            x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=True)(x)
            #x = Dropout(0.1)(x, training=True)
        else:
            x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same', use_bias=True)(x)  # this matches fractionally stided with stride 1/2
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        # x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)  # this matches fractionally stided with stride 1/2
        x = BatchNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def Discriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.data_shapeB)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_img, 16, False, 2)
        # Layer 2
        x = self.ck(x, 32, True, 2)
        # Layer 3
        x = self.ck(x, 64, True, 2)
        # Layer 4
        x = self.ck(x, 128, True, 2)
        # Layer 5
        x = self.ck(x, 256, True, 2)
        # Layer 6
        x = self.ck(x, 512, True, 2)
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)
        # Output layer
        # x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same', use_bias=True)(x)
        # x = Activation('sigmoid')(x)
        return Model(inputs=input_img, outputs=x, name=name)

    def Generator(self, name=None):
        # input_img = Input(shape=self.data_shapeA)
        # x = ReflectionPadding2D((3, 3))(input_img)
        # x = self.c7Ak(x, 32)
        #
        # # Layer 2-3: Downsampling
        # x = self.dk(x, 64)
        # x = self.dk(x, 128)
        #
        # # Layers 4-12: Residual blocks
        # for _ in range(4, 7):
        #     x = self.Rk(x)
        #
        # # Layer 13:14: Upsampling
        # x = self.uk(x, 64)
        # x = self.uk(x, 32)
        #
        # # Layer 15: Output
        # x = ReflectionPadding2D((3, 3))(x)
        # x = Conv2D(filters=self.data_shapeA[2]-1, kernel_size=7, strides=1, padding='valid', use_bias=True)(x)
        # x = Activation('tanh')(x)
        #
        # return Model(inputs=input_img, outputs=x, name=name)

        img_input = Input(shape=self.data_shapeA)

        # Block 1
        x = Conv2D(16, (3, 3), padding='same', name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        block_1_out = Activation('relu')(x)

        # x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
        # x = BatchNormalization()(x)
        # block_1_out = Activation('relu')(x)

        x = MaxPooling2D()(block_1_out)

        # Block 2
        x = Conv2D(32, (3, 3), padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        block_2_out = Activation('relu')(x)

        # x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
        # x = BatchNormalization()(x)
        # block_2_out = Activation('relu')(x)

        x = MaxPooling2D()(block_2_out)

        # Block 3
        x = Conv2D(64, (3, 3), padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        block_3_out = Activation('relu')(x)

        # x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        #
        # x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
        # x = BatchNormalization()(x)
        # block_3_out = Activation('relu')(x)

        x = MaxPooling2D()(block_3_out)

        # Block 4
        x = Conv2D(128, (3, 3), padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        block_4_out: Optional[Any] = Activation('relu')(x)

        # x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        #
        # x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
        # x = BatchNormalization()(x)
        # block_4_out = Activation('relu')(x)

        x = MaxPooling2D()(block_4_out)

        # Block 5
        x = Conv2D(256, (3, 3), padding='same', name='block5_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3, 3), padding='same', name='block5_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        #
        # x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)

        # UP 1
        x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = concatenate([x, block_4_out])
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # x = Conv2D(512, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)

        # UP 2
        x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = concatenate([x, block_3_out])
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # x = Conv2D(256, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)

        # UP 3
        x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = concatenate([x, block_2_out])
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # x = Conv2D(128, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)

        # UP 4
        x = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = concatenate([x, block_1_out])
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # x = Conv2D(64, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)

        # last conv
        x = Conv2D(self.data_shapeA[2], (3, 3), activation='tanh', padding='same')(x)

        # model = Model(img_input, x)
        # model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
        #               loss='categorical_crossentropy',
        #               metrics=[dice_coef])

        return Model(inputs=img_input, outputs=x, name=name)






    def train(self, train_A_dir, normalization_factor_A, train_B_dir, normalization_factor_B, models_dir, batch_size=10,
              epochs=200, cycle_loss_type='L1', use_resize_convolution=False, use_supervised_learning=True,
              output_sample_flag=True, output_sample_dir='./output_sample', output_sample_channels=1, dropout_rate=0):
        self.batch_size = batch_size
        self.epochs = epochs
        self.decay_epoch = self.epochs//2 # the epoch where linear decay of the learning rates starts
        self.cycle_loss_type = cycle_loss_type
        self.use_resize_convolution = use_resize_convolution
        self.use_supervised_learning = use_supervised_learning
        self.dropout_rate = dropout_rate
        # Data dir
        self.train_A_dir = train_A_dir
        self.train_B_dir = train_B_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        self.models_dir = models_dir
        self.train_A = load_data(join(self.train_A_dir, '10.nii'), normalization_factor_A)
        self.train_B = load_data(join(self.train_B_dir, '10.nii'), normalization_factor_B)
        self.data_shapeA = self.train_A.shape[1:4]
        self.data_shapeB = self.train_B.shape[1:4]
        # self.data_num = self.train_A.shape[0]
        self.loop_num = len(os.listdir(self.train_A_dir))
        # print('Number of epochs: {}, number of loops per epoch: {}'.format(self.epochs, self.loop_num))
        self.create_discriminator_and_generator()

        # Image pools used to update the discriminators
        # self.synthetic_A_pool = ImagePool(self.synthetic_pool_size)
        # self.synthetic_B_pool = ImagePool(self.synthetic_pool_size)

        GGloss = []
        PPixelloss = []
        DDloss = []
        start_time = time.time()
        print("Dropout rate: {}".format(dropout_rate))
        print('Training ...')
        for epoch_i in range(self.epochs):
            # Update learning rates
            # if epoch_i > self.decay_epoch:
            #     self.update_lr(self.D_A, decay_D)
            #     self.update_lr(self.D_B, decay_D)
            #     self.update_lr(self.G_model, decay_G)
            # random_indices = np.random.permutation(self.data_num)
            Gloss = 0
            Pixelloss = 0
            Dloss = 0
            j = 0
            for loop_j in os.listdir(self.train_A_dir):
                # training data batches
                if self.use_supervised_learning:
                    # key = loop_j.split('_', 1)[1]
                    train_A_batch = load_data(join(self.train_A_dir, loop_j), normalization_factor_A)
                    train_B_batch = load_data(join(self.train_B_dir, loop_j), normalization_factor_B)
                    # train_B_batch = load_data(join(self.train_B_dir, 'SpOrange_'+key), normalization_factor_B)

                    label_shape = (train_B_batch.shape[0],) + self.D_B.output_shape[1:]
                    ones = np.ones(shape=label_shape)
                    zeros = ones * 0
                else:
                    key = loop_j.split('_', 1)[1]
                    train_A_batch = load_data(join(self.train_A_dir, loop_j), normalization_factor_A)
                    train_B_batch = load_data(join(self.train_B_dir, 'SpOrange_'+key), normalization_factor_B)

                    label_shape = (train_B_batch.shape[0],) + self.D_B.output_shape[1:]
                    ones = np.ones(shape=label_shape)
                    zeros = ones * 0
                # Synthetic data for training data batches
                synthetic_B_batch = self.G_A2B.predict(train_A_batch)
                # synthetic_A_batch = self.G_B2A.predict(train_B_batch)
                # synthetic_A_batch = self.synthetic_A_pool.query(synthetic_A_batch)
                # synthetic_B_batch = self.synthetic_B_pool.query(synthetic_B_batch)

                # Train Discriminator
                # DA_loss_train = self.D_A.train_on_batch(x=train_A_batch, y=ones)
                DB_loss_train = self.D_B.train_on_batch(x=train_B_batch, y=ones)
                # DA_loss_synthetic = self.D_A.train_on_batch(x=synthetic_A_batch, y=zeros)
                DB_loss_synthetic = self.D_B.train_on_batch(x=synthetic_B_batch, y=zeros)
                # D_loss = DA_loss_train + DA_loss_synthetic + DB_loss_train + DB_loss_synthetic

                # target_data = [train_B_batch]
                target_data = [ones]
                # target_data.append(ones)
                # target_data.append(ones)
                if self.use_supervised_learning:
                    # target_data.append(train_A_batch)
                    target_data.append(train_B_batch)
                # Train Generator
                G_loss = self.G_model.train_on_batch(x=[train_A_batch], y=target_data)
                self.print_info(start_time, epoch_i,  G_loss,  DB_loss_train + DB_loss_synthetic)
                Gloss = Gloss + G_loss[1]
                Pixelloss = Pixelloss + G_loss[2]
                Dloss = Dloss + DB_loss_train + DB_loss_synthetic
                # self.print_info(start_time, epoch_i,  D_loss, G_loss, DA_loss_train + DA_loss_synthetic,\
                #                 DB_loss_train + DB_loss_synthetic)
                # if (output_sample_flag):
                #     if (loop_j + 1) % 5 == 0:
                #         first_row = np.rot90(train_A_batch[0, :, :, 0])  # training data A
                #         second_row = np.rot90(train_B_batch[0, :, :, 0])  # training data B
                #         third_row = np.rot90(synthetic_B_batch[0, :, :, 0])  # synthetic data B
                #         if output_sample_channels > 1:
                #             for channel_i in range(output_sample_channels - 1):
                #                 first_row = np.append(first_row, np.rot90(train_A_batch[0, :, :, channel_i + 1]),
                #                                       axis=1)
                #                 second_row = np.append(second_row, np.rot90(train_B_batch[0, :, :, channel_i + 1]),
                #                                        axis=1)
                #                 third_row = np.append(third_row, np.rot90(synthetic_B_batch[0, :, :, channel_i + 1]),
                #                                       axis=1)
                #         output_sample = np.append(np.append(first_row, second_row, axis=0), third_row, axis=0)
                #         # toimage(output_sample, cmin=-1, cmax=1).save(output_sample_dir)
                #         np.save(output_sample_dir, output_sample)
                j = j + 1
            if (epoch_i + 1) % 1 == 0:
                print('Gloss_epoch_%.3d: %f' % (epoch_i+1, Gloss/j))
                print('Pixelloss_epoch_%.3d: %f' % (epoch_i+1, Pixelloss/j))
                print('Dloss_epoch_%.3d: %f' % (epoch_i+1, Dloss/j))
                GGloss.append(Gloss/j)
                PPixelloss.append(Pixelloss/j)
                DDloss.append(Dloss/j)
                self.save_model(epoch_i)
            print("\u001b[12B")
            print("\u001b[1000D")
            print('Done')
        np.savetxt(join(models_dir, 'Gloss.txt'), np.array(GGloss))
        np.savetxt(join(models_dir, 'Pixelloss.txt'), np.array(PPixelloss))
        np.savetxt(join(models_dir, 'Dloss.txt'), np.array(DDloss))

    def synthesizeX2Y(self, G_X2Y_dir, test_X_dir, test_Y_dir, normalization_factor_X, synthetic_Y_dir, normalization_factor_Y,
                      use_resize_convolution=False, dropout_rate=0):

        i = 0
        val_totalloss = 0
        patchList = os.listdir(test_X_dir)
        patchList.sort()
        patchList.sort(key=lambda x: x[:-4])
        for loop_i in patchList:
            valPixelloss = 0
            # yloop_i = 'SpOrange' + loop_i.split('x')[1]
            # test_X_img=np.load(test_X_dir)
            test_X = load_data(join(test_X_dir, loop_i), normalization_factor_X)
            test_Y = load_data(join(test_Y_dir, loop_i), normalization_factor_Y)
            self.data_shapeA = test_X.shape[1:4]
            self.data_num = test_X.shape[0]
            self.use_resize_convolution = use_resize_convolution
            self.dropout_rate = dropout_rate
            print('Synthesizing ...')
            # print("Dropout rate: {}".format(dropout_rate))
            self.G_X2Y = self.Generator(name='G_X2Y')
            self.G_X2Y.load_weights(G_X2Y_dir)
            synthetic_Y = self.G_X2Y.predict(test_X)
            from sklearn.metrics import mean_squared_error
            for j in range(synthetic_Y.shape[0]):
                val_loss = mean_squared_error(np.squeeze(test_Y[j]), np.squeeze(synthetic_Y[j]))
                valPixelloss = valPixelloss + val_loss
            # valPixelloss = tf.reduce_mean(tf.math.squared_difference(synthetic_Y, test_Y))
            valPixelloss = valPixelloss/synthetic_Y.shape[0]
            print(valPixelloss)
            val_totalloss = val_totalloss + valPixelloss
            synthetic_Y = denormalize_data(synthetic_Y, normalization_factor_Y)
            synthetic_Y[synthetic_Y < 0] = 0
            print("max value:", np.max(synthetic_Y))
            print("min value:", np.min(synthetic_Y))
            print("mean value:", np.mean(synthetic_Y))
            synthetic_Y = synthetic_Y[:, 0:test_X.shape[1], 0:test_X.shape[2], :]  # Remove padded zeros
            S_Y = sitk.GetImageFromArray(synthetic_Y)
            sitk.WriteImage(S_Y, '/media/aiilab/Experiments/TanHanbo/GAN/outputs_DAPI_SpGreen_GAN/synthetic/synthetic_epoch_50/4.nii.gz')
            # np.save(join(synthetic_Y_dir, 'synSpOrange_05_%03d.npy' % i), synthetic_Y.astype(dtype=np.float16))
            # np.save(join(synthetic_Y_dir, 'synSpOrange_B16_%03d.npy' % i), synthetic_Y.astype(dtype=np.float16))

            i = i+1
            print('Done\n')
        valloss = val_totalloss/i
        return valloss

    def synthesizeY2X(self, G_Y2X_dir, test_Y_dir, normalization_factor_Y, synthetic_X_dir, normalization_factor_X,
                      use_resize_convolution=False, dropout_rate=0):
        test_Y_img = np.load(test_Y_dir)
        test_Y = load_data(test_Y_dir, normalization_factor_Y)
        self.data_shape = test_Y.shape[1:4]
        self.data_num = test_Y.shape[0]
        self.use_resize_convolution = use_resize_convolution
        self.dropout_rate = dropout_rate
        print('Synthesizing ...')
        print("Dropout rate: {}".format(dropout_rate))
        self.G_Y2X = self.Generator(name='G_Y2X')
        self.G_Y2X.load_weights(G_Y2X_dir)
        synthetic_X = self.G_Y2X.predict(test_Y)
        synthetic_X = denormalize_data(synthetic_X, normalization_factor_X)
        synthetic_X[synthetic_X < 0] = 0
        synthetic_X = synthetic_X[:, 0:test_Y_img.shape[1], 0:test_Y_img.shape[2], :]  # Remove padded zeros
        np.save(synthetic_X_dir, synthetic_X)
        print('Done\n')

    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))
        return loss

    def cycle_loss(self, y_true, y_pred):
        if self.cycle_loss_type == 'L1':
            # L1 norm
            loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss

    def get_lr_linear_decay_rate(self):
        updates_per_epoch_D = 2 * self.data_num
        updates_per_epoch_G = self.data_num
        denominator_D = (self.epochs - self.decay_epoch) * updates_per_epoch_D
        denominator_G = (self.epochs - self.decay_epoch) * updates_per_epoch_G
        decay_D = self.lr_D / denominator_D
        decay_G = self.lr_G / denominator_G
        return decay_D, decay_G

    def update_lr(self, model, decay):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        K.set_value(model.optimizer.lr, new_lr)

    def print_info(self, start_time, epoch_i, G_loss, DB_loss):
        print("\n")
        print("Epoch               : {:d}/{:d}{}".format(epoch_i + 1, self.epochs, "                         "))
        # print("Loop                : {:d}/{:d}{}".format(loop_j + 1, self.loop_num, "                         "))
        # print("D_loss              : {:5.4f}{}".format(D_loss, "                         "))
        print("Total_loss              : {:5.4f}{}".format(G_loss[0], "                         "))
        print("G_loss              : {:5.4f}{}".format(G_loss[1], "                         "))
        print("Pixelloss              : {:5.4f}{}".format(G_loss[2], "                         "))
        # print("DA_loss             : {:5.4f}{}".format(DA_loss, "                         "))
        print("DB_loss             : {:5.4f}{}".format(DB_loss, "                         "))
        # passed_time = (time.time() - start_time)
        # loops_finished = epoch_i * self.loop_num
        # loops_total = self.epochs * self.loop_num
        # loops_left = loops_total - loops_finished
        # remaining_time = (passed_time / (loops_finished + 1e-5) * loops_left)
        # passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
        # remaining_time_string = str(datetime.timedelta(seconds=round(remaining_time)))
        # print("Time passed         : {}{}".format(passed_time_string, "                         "))
        # print("Time remaining      : {}{}".format(remaining_time_string, "                         "))
        print("\u001b[13A")
        print("\u001b[1000D")
        sys.stdout.flush()

    def save_model(self, epoch_i):
        models_dir_epoch_i = os.path.join(self.models_dir,
                                          '{}_weights_epoch_{}.hdf5'.format(self.G_A2B.name, epoch_i + 1))
        self.G_A2B.save_weights(models_dir_epoch_i)
        # models_dir_epoch_i = os.path.join(self.models_dir,
        #                                   '{}_weights_epoch_{}.hdf5'.format(self.G_B2A.name, epoch_i + 1))
        # self.G_B2A.save_weights(models_dir_epoch_i)
        #


    # def synthesizeX2Y(self, G_X2Y_dir, test_X_dir, normalization_factor_X, synthetic_Y_dir, normalization_factor_Y, use_resize_convolution=False, dropout_rate=0):
    #     i = 0
    #     for loop_i in os.listdir(test_X_dir):
    #         # test_X_img=np.load(test_X_dir)
    #         test_X = load_data(join(test_X_dir, loop_i), normalization_factor_X)
    #         self.data_shapeA = test_X.shape[1:4]
    #         self.data_num = test_X.shape[0]
    #         self.use_resize_convolution = use_resize_convolution
    #         self.dropout_rate = dropout_rate
    #         print('Synthesizing ...')
    #         print("Dropout rate: {}".format(dropout_rate))
    #         self.G_X2Y = self.Generator(name='G_X2Y')
    #         self.G_X2Y.load_weights(G_X2Y_dir)
    #         synthetic_Y = self.G_X2Y.predict(test_X)
    #         synthetic_Y = denormalize_data(synthetic_Y, normalization_factor_Y)
    #         synthetic_Y[synthetic_Y < 0] = 0
    #         print("max value:", np.max(synthetic_Y))
    #         print("min value:", np.min(synthetic_Y))
    #         print("mean value:", np.mean(synthetic_Y))
    #         # synthetic_Y = synthetic_Y[:, 0:test_X.shape[1], 0:test_X.shape[2], :]  # Remove padded zeros
    #         np.save(join(synthetic_Y_dir, 'synSpOrange_05_%03d.npy' % i), synthetic_Y.astype(dtype=np.float16))
    #         i = i+1
    #         print('Done\n')
    # #
    # def synthesizeY2X(self, G_Y2X_dir, test_Y_dir, normalization_factor_Y, synthetic_X_dir, normalization_factor_X, use_resize_convolution=False, dropout_rate=0):
    #     test_Y_img=np.load(test_Y_dir)
    #     test_Y = load_data(test_Y_dir, normalization_factor_Y)
    #     self.data_shape = test_Y.shape[1:4]
    #     self.data_num = test_Y.shape[0]
    #     self.use_resize_convolution = use_resize_convolution
    #     self.dropout_rate = dropout_rate
    #     print('Synthesizing ...')
    #     print("Dropout rate: {}".format(dropout_rate))
    #     self.G_Y2X = self.Generator(name='G_Y2X')
    #     self.G_Y2X.load_weights(G_Y2X_dir)
    #     synthetic_X = self.G_Y2X.predict(test_Y)
    #     synthetic_X = denormalize_data(synthetic_X, normalization_factor_X)
    #     synthetic_X[synthetic_X<0] = 0
    #     synthetic_X = synthetic_X[:, 0:test_Y_img.shape[1], 0:test_Y_img.shape[2],:]  # Remove padded zeros
    #     np.save(synthetic_X_dir, synthetic_X)
    #     print('Done\n')
    #
    # def lse(self, y_true, y_pred):
    #     loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))
    #     return loss
    #
    # def cycle_loss(self, y_true, y_pred):
    #     if self.cycle_loss_type == 'L1':
    #         # L1 norm
    #         loss = tf.reduce_mean(tf.abs(y_pred - y_true))
    #     return loss
    #
    # def get_lr_linear_decay_rate(self):
    #     updates_per_epoch_D = 2 * self.data_num
    #     updates_per_epoch_G = self.data_num
    #     denominator_D = (self.epochs - self.decay_epoch) * updates_per_epoch_D
    #     denominator_G = (self.epochs - self.decay_epoch) * updates_per_epoch_G
    #     decay_D = self.lr_D / denominator_D
    #     decay_G = self.lr_G / denominator_G
    #     return decay_D, decay_G
    #
    # def update_lr(self, model, decay):
    #     new_lr = K.get_value(model.optimizer.lr) - decay
    #     if new_lr < 0:
    #         new_lr = 0
    #     K.set_value(model.optimizer.lr, new_lr)
    #
    # def print_info(self, start_time, epoch_i, loop_j, D_loss_train, D_loss_synthetic, G_loss, Content_loss):
    #     print("\n")
    #     print("Epoch               : {:d}/{:d}{}".format(epoch_i + 1, self.epochs, "                         "))
    #     # print("Loop                : {:d}/{:d}{}".format(loop_j + 1, self.loop_num, "                         "))
    #     print("D_loss              : {:5.4f}{}".format(D_loss_train, "                         "))
    #     print("D_loss              : {:5.4f}{}".format(D_loss_synthetic, "                         "))
    #     print("G_loss              : {:5.4f}{}".format(G_loss, "                         "))
    #     # print("reconstruction_loss : {:5.4f}{}".format(G_loss[3]+ G_loss[4], "                         "))
    #     # print("DA_loss             : {:5.4f}{}".format(DA_loss, "                         "))
    #     print("Content_loss             : {:5.4f}{}".format(Content_loss, "                         "))
    #     passed_time = (time.time() - start_time)
    #     loops_finished = epoch_i * self.loop_num + loop_j
    #     loops_total = self.epochs * self.loop_num
    #     loops_left = loops_total - loops_finished
    #     remaining_time = (passed_time / (loops_finished + 1e-5) * loops_left)
    #     passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
    #     remaining_time_string = str(datetime.timedelta(seconds=round(remaining_time)))
    #     print("Time passed         : {}{}".format(passed_time_string, "                         "))
    #     print("Time remaining      : {}{}".format(remaining_time_string, "                         "))
    #     print("\u001b[13A")
    #     print("\u001b[1000D")
    #     sys.stdout.flush()
    #
    # def save_model(self, epoch_i):
    #     models_dir_epoch_i = os.path.join(self.models_dir, '{}_weights_epoch_{}.hdf5'.format(self.G_A2B.name, epoch_i+1))
    #     self.G_A2B.save_weights(models_dir_epoch_i)
    #     models_dir_epoch_i = os.path.join(self.models_dir, '{}_weights_epoch_{}.hdf5'.format(self.G_B2A.name, epoch_i+1))
    #     self.G_B2A.save_weights(models_dir_epoch_i)