'''
 * User: Hojun Lim
 * Date: 2020-04-01
'''

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as datasets
import os
from PIL import Image
import matplotlib.pyplot as plt

test_data_class10_dir = os.path.join(os.getcwd(),'dataset\\class10')
def get_data_loader(data_dir= test_data_class10_dir, batch_size=1, train = True, valid = True):
    """
    Define the way we compose the batch dataset including the augmentation for increasing the number of data
    and return the augmented batch-dataset

    :param data_dir: root directory where the either train or test dataset is
    :param batch_size: size of the batch
    :param train: true if current phase is training, else false
    :return: augmented batch dataset
    """

    if valid: # when load data for validation
        data_dir = os.path.join(data_dir, 'validation')
    else: # when load data for train or test
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

    # ImageFloder and dataloader, with root directory and defined transformation methods, for mini-batch as well as data augmentation
    if valid: # when load data for validation
        data = torchvision.datasets.ImageFolder(root=data_dir, transform=transform['test'])
    else: # when load data for train or test
        data = torchvision.datasets.ImageFolder(root=data_dir, transform=transform['train'] if train else 'test')

    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader


def exercise_SubsetRandomSampler():
    """
    function to practice the subsetRandomsampler in order to build validationset
    """
    # Shuffle the indices
    indices = np.arange(0,60000)
    np.random.shuffle(indices)

    # Build the train loader
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', download=True, train=True,
                                                              transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                                               batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[:55000]))

    # Build the validation loader
    val_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', download=True, train=True,
                                                            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                                             batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[-5000:]))

    return train_loader, val_loader

# The way to get one batch from the data_loader
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    data_loader = get_data_loader()
    print(iter(data_loader))
    print(len(iter(data_loader)))
#    for i in range(10):
#        batch_x, batch_y = next(iter(data_loader))
#        print(np.shape(batch_x), batch_y)

    i = 0
    for x,y in data_loader:
        if i == 0:
            a = x
            #print(x.tolist()[:1])

        i+=1

    i = 0
    for x,y in data_loader:
        if i == 0:
            b = x
            #print(x.tolist()[:1])
        i+=1

    c = torchvision.transforms.ToPILImage("RGB")(a[0])
    c.show()
    #d = torchvision.transforms.ToPILImage("RGB")(b[0])
    #d.show()
    torch.T

class FoodDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform): # from torch.utils.data.Dataset
        """

        :param root: the directory where the dataset is
        :param transform: transformation that is needed to be applied to data before it is loaded by DataLoader
        """
        self.root = root
        self.transform = transform
        self.x_data, self.y_data = self.get_entire_dataset(root, transform)


        return

    def __len__(self): # from torch.utils.data.Dataset
        """

        :return: the length of dataset, the number of data in dataset
        """
        return len(self.x)

    def __getitem__(self, idx): # from torch.utils.data.Dataset
        """

        :param idx: index of data and corresponding label that we want to retrieve from dataset
        :return: x_data[idx], y_data[idx]
        """
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x, y

    def get_entire_dataset(self, root, transfrom):

        # Open and load all images in the root directory, then return those data being transformed by the given argument transform.
        return

