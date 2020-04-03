'''
 * User: Hojun Lim
 * Date: 2020-04-01
'''

import torch.nn as nn

import torch
import torchvision

from model import *
from data_loader import * 
import torch.optim as optim
import torch.nn.functional as F

#TODO add comments
def train(learning_rate, batch_size, dir_dataset, device, model_name, num_classes, epoch):
    data_loader = get_data_loader(data_dir=dir_dataset, batch_size=batch_size, train=True)
    model = get_pretrained_network(model_name=model_name, num_classes=num_classes,use_trained=True,device=device)

    # define the adam optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    loss_per_epoch = []

    for current_epoch in range(epoch):

        epoch_loss = 0

        # variable to save the number of correct model prediction per epoch
        crt_prd_per_epoch = 0
        total_prd = 0

        for input, label in data_loader:

            # move tensor from cpu to gpu
            input = input.to(device)
            label = label.to(device)

            # clear the accumulated gradients from last batch
            optimizer.zero_grad()

            # send the input(tensor) to the GPU if gpu is available
            model_output = model(input)
            model_prediction = torch.argmax(model_output,1)
            # define loss function as cross entropy
            loss_fn = F.cross_entropy(model_output, label)

            # calculate the derivatives with respect to parameters
            loss_fn.backward()
            # update the parameters
            optimizer.step()

            # calculate the average loss of the current epoch
            epoch_loss += batch_size*loss_fn.item()

            # variables to calculate the accuracy
            crt_prd_per_epoch += torch.sum(model_prediction.eq(label.to(device))).item()
            total_prd += batch_size

            torch.cuda.empty_cache()


            ########
        loss_per_epoch.append(epoch_loss)
        print('Epoch {} {}: {:3f}'.format(current_epoch+1, 'loss', epoch_loss))
        print('Epoch {} {}: {:3f}'.format(current_epoch+1, 'accuracy', 100* crt_prd_per_epoch/total_prd))

        #print('update', model.fc.weight)

    #TODO Save log and visualize the loss graphs
    return




