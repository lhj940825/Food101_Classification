'''
 * User: Hojun Lim
 * Date: 2020-04-01
'''

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os


test_data_class10_dir = os.path.join(os.getcwd(),'dataset\\class10')
def get_data_loader(data_dir= test_data_class10_dir, batch_size=6, train = True):
    """
    Define the way we compose the batch dataset including the augmentation for increasing the number of data
    and return the augmented batch-dataset

    :param data_dir: root directory where the either train or test dataset is
    :param batch_size: size of the batch
    :param train: true if current phase is training, else false
    :return: augmented batch dataset
    """

    data_dir = os.path.join(data_dir, 'train' if train==True else 'test')
    # define how we augment the data for composing the batch-dataset in train and test step
    transform = {
        'train': transforms.Compose([
            transforms.Resize([224,224]), # Resizing the image as the VGG only take 224 x 244 as input size
            transforms.RandomHorizontalFlip(), # Flip the data horizontally
            #TODO if it is needed, add the random crop
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ]),
        'test': transforms.Compose([
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    }

    # ImageFloder with root directory and defined transformation methods for batch as well as data augmentation
    data = torchvision.datasets.ImageFolder(root=data_dir, transform=transform['train'] if train else 'test')
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader


# The way to get one batch from the data_loader
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    data_loader = get_data_loader()
    print(len(data_loader))
    for i in range(10):
        batch_x, batch_y = next(iter(data_loader))
        print(np.shape(batch_x), batch_y)





