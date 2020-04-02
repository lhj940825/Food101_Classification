'''
 * User: Hojun Lim
 * Date: 2020-04-01
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class Network(nn.Module):

    def __init__(self):
        return

    def forward(self, x):

        return




def get_pretrained_network(model_name, num_classes, use_trained=True, device="cpu"):
    """
    Return the respective pretrained model with modifed fully connected layer corresponding the number of classes


    :param model_name: name of the model which we want to load
    :param num_classes: number of classes to classify
    :param use_trained: load the pretrained model if True
    :param device: GPU or CPU
    :return: loaded pretrained model(ResNet or VGG)
    """

    # when the resNet is expected to be returned
    if model_name == "resnet":
        model = models.resnet34(pretrained=use_trained)
        set_parameter_requires_grad(model, feature_extracting=True)

        ## In order to finetune the model, we need to update the last layer(fully connect layer) so that the model becomes our classification task-specific
        num_features = model.fc.in_features
        # update the fully connected layer
        model.fc = nn.Linear(num_features, num_classes)
        model.to(device)

        return model
    elif model_name == "vgg":
        model = models.vgg16(pretrained=use_trained)
        set_parameter_requires_grad(model, feature_extracting=True)

        #model.classifier[6] refers to the fully connected layer of VGG
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
        model.to(device)

        return model
    else:
        raise Exception('Wrong model type, it should be either resnet or vgg')

def set_parameter_requires_grad(model: nn.Module, feature_extracting=True):
    """
    reference:https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

    This helper function sets the '.requires_grad' attribute of the parameters in the given model to False.
    By default, this attributes are set True when we load a pretrained model. This makes sense if we want to
    finetune or train the model from scratch. However, if our aim with this model is to do feature extraction
    and to compute gradients for the newly initialized layer then it is desirable to set the '.required_grad' attribute False.

    :param model:
    :param feature_extracting:
    """

    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

    return


# In order to test out whether the pretrained models are loaded appropriately
if __name__ == '__main__':
    model = get_pretrained_network('vgg', 10,True,"cuda:0")
    print(model)








