# Import required libraries
import sys
sys.path.append('/share/home/scz6051/Experiments/HRNet/yx3/utils')
import torch
import numpy as np
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.unetdataset import Unet_Loader
from configer import Configer
from model import hrnet


def train_net(net, train_path, val_path, epoch=20, batch_size=32, lr=1e-5):
    # Create dataset loaders for training and validation
    train_dataset = Unet_Loader(train_path, training=True)
    train_loader = DataLoader(dataset=train_dataset,
                              num_workers=6,
                              batch_size=batch_size,
                              shuffle=True)

    val_dataset = Unet_Loader(val_path, training=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=32,
                            shuffle=False)
    # Set up optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    # Set up loss function
    Loss_function = nn.MSELoss()
    best_loss = float('inf')
    training_loss = []
    Val_loss = []

    # Training loop
    for i in range(epoch):
        net.train()
        train_sumloss = 0
        with tqdm(total=train_loader.__len__()) as pbar:
            for image, label in train_loader:
                optimizer.zero_grad()
                image = image.cuda().float()
                label = label.cuda().float()
                # label = sigmoid(label)

                pred = net(image)
                # pred = sigmoid(pred)

                loss = Loss_function(pred, label)
                loss.backward()
                train_sumloss += loss

                optimizer.step()
                pbar.set_postfix(loss=float(loss.cpu()), epoch=i)
                pbar.update(1)
        # Validation loop
        if i % 1 == 0:
            with torch.no_grad():
                net.eval()
                pre_stack = []
                label_stack = []
                val_sumloss = 0
                for image, label in tqdm(val_loader, total=len(val_loader), desc=f'Epoch {i + 1}/{epoch}'):
                    image = image.cuda().float()
                    label = label.cuda().float()
                    # label = sigmoid(label)

                    pred = net(image)
                    # pred = sigmoid(pred)

                    val_loss = Loss_function(pred, label)
                    val_sumloss += val_loss

                    pred = pred.cpu().numpy().astype(np.double)
                    label = label.cpu().numpy().astype(np.double)
                    pre_stack.append(pred)
                    label_stack.append(label)

                if val_sumloss < best_loss:
                    best_loss = val_sumloss
                    torch.save(net.state_dict(), 'Ablation_stage1.pth')
                    print('save model')

        # Calculate and store average losses
        trainloss = train_sumloss / len(train_loader)
        valloss = val_sumloss / len(val_loader)
        training_loss.append(trainloss)
        Val_loss.append(valloss)
        print('train_sumloss:', train_sumloss)
        print('train_loss:', trainloss)
        print('val_sumloss:', val_sumloss)
        print('val_loss:', valloss)
    # Print and plot training and validation losses
    print("fluo5 DAPI+SpGreen Training_loss:", training_loss)
    print("fluo5 DAPI+SpGreen Val_loss:", Val_loss)

    epochs = range(1, len(training_loss)+1)
    plt.plot(epochs, training_loss.cpu(), 'y', label='Training loss')
    plt.plot(epochs, Val_loss.cpu(), 'r', label='Val loss')
    plt.title('Training and Val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()










if __name__ == "__main__":
    # Load the configuration
    configer = Configer(configs='utils/H_48_D_4.json')
    net = hrnet.HRNet_W48(configer)
    net = net.cuda()

    # Set up paths for training and validation data
    train_path = "/share/home/scz6051/Experimental_Data/HRNet_data/Patches_DAPI_SpGreen/ComplexTr/"
    val_path = "/share/home/scz6051/Experimental_Data/HRNet_data/Patches_DAPI_SpGreen/ComplexVal/"

    # Train the model
    train_net(net, train_path, val_path)
