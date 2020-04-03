'''
 * User: Hojun Lim
 * Date: 2020-04-01
'''


import torch
from utils import *
from train import *


def main():
    torch.multiprocessing.freeze_support()
    #TODO yml parsing, class10 trainset increase
    ###parsed_data = parse_yml('config.yml')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_data_class10_dir = os.path.join(os.getcwd(),'dataset')
    task='class10'
    num_classes = 10
    model_name = 'resnet'
    dir_dataset =os.path.join(test_data_class10_dir,task)
    epoch = 20
    batch_size = 10
    #TODO dir_data and type of task come from parsed_data
    train(learning_rate=0.001, batch_size=batch_size, dir_dataset=dir_dataset, device=device, model_name=model_name, num_classes = num_classes, epoch=epoch)


if __name__ == "__main__":
    main()
