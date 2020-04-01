'''
 * User: Hojun Lim
 * Date: 2020-04-01
'''

import torch


def is_gpu_available():
    """
    check whether the training goes with GPU or not
    :return: true or false
    """
    return torch.cuda.is_available()
