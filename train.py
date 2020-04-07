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
from torch.utils.tensorboard import SummaryWriter
from utils import *

#TODO add comments
def train(learning_rate, batch_size, dir_dataset, device, model_name, num_classes, epoch, log_dir):
    data_loader = get_data_loader(data_dir=dir_dataset, batch_size=batch_size, train=True, valid=False)
    model = get_pretrained_network(model_name=model_name, num_classes=num_classes,use_trained=True,device=device)


    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate) # define the adam optimizer
    writer = SummaryWriter(log_dir=log_dir) # define summaryWriter to make logs for tensorboard(visualization)

    loss_per_epoch = []


    for current_epoch in range(epoch):
        epoch_loss = 0


        crt_prd_per_epoch = 0 # variable to save the number of correct model prediction per epoch
        cnt_correct_prediction = 0 # variable to count the number of correct prediction from model
        total_prd = 0 # variable to count the total number of prediction produced by the model

        for i, data in enumerate(data_loader):
            model.train() # set model in train mode

            training_loss = 0 # variable to store the loss value
            input , label = data

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

            training_loss += loss_fn.item() # update the loss
            cnt_correct_prediction += torch.sum(torch.eq(model_prediction, label))
            total_prd += batch_size

            # make a log for every 10 iterations
            if i%10 == 9:
                training_accuracy = 100*cnt_correct_prediction/total_prd
                global_step = current_epoch*len(data_loader)+i
                print('training loss', training_loss/10)
                writer.add_scalar('training loss', training_loss/10, global_step=global_step)
                writer.add_scalar('training_accuracy', training_accuracy, global_step=global_step)
                writer.add_image('input image', input[0])

                # compute the validaion loss and make a log
                validate(model=model, batch_size=batch_size, dir_dataset=dir_dataset, device=device, global_step=global_step, writer=writer)

            torch.cuda.empty_cache()
    
    torch.save(model,'.\\logs\\model_resnet.pth.tar')
    writer.close()
    return model

def validate(model: torch.nn.modules, batch_size, dir_dataset, device, global_step, writer: SummaryWriter):
    model.eval() # set the model evaluation mode
    data_loader = get_data_loader(data_dir=dir_dataset, batch_size=batch_size,train=False, valid=True)

    validation_loss = 0
    with torch.no_grad(): # without gradient calculation
        print(len(data_loader))
        for input, label in data_loader:

            # move tensor from cpu to gpu
            input = input.to(device)
            label = label.to(device)

            model_output = model(input)
            model_prediction = torch.argmax(model_output, 1)
            print(model_prediction, label)
            loss_fn = F.cross_entropy(model_output, label)

            validation_loss+=loss_fn

        validation_loss = validation_loss/len(data_loader)
        writer.add_scalar('validation_loss', validation_loss, global_step=global_step)

    return




