'''
 * User: Hojun Lim
 * Date: 2020-04-01
'''

import torch
import shutil
import os
import numpy as np
import argparse

def is_gpu_available():
    """
    check whether the training goes with GPU or not
    :return: true or false
    """
    return torch.cuda.is_available()



def get_files_from_folder(path):

    files = os.listdir(path)
    return np.asarray(files)


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset divider")
    parser.add_argument("--data_path", required=True,
                        help="Path to data")
    parser.add_argument("--test_data_path_to_save", required=True,
                        help="Path to test data where to save")
    parser.add_argument("--train_ratio", required=True,
                        help="Train ratio - 0.7 means splitting data in 70 % train and 30 % test")
    return parser.parse_args()

def split_dataset(path_to_data, path_to_test_data, train_ratio):
    # get dirs
    _, dirs, _ = next(os.walk(path_to_data))

    # calculates how many train data per class
    data_counter_per_class = np.zeros((len(dirs)))
    for i in range(len(dirs)):
        path = os.path.join(path_to_data, dirs[i])
        files = get_files_from_folder(path)
        data_counter_per_class[i] = len(files)
    test_counter = np.round(data_counter_per_class * (1 - train_ratio))

    # transfers files
    for i in range(len(dirs)):
        path_to_original = os.path.join(path_to_data, dirs[i])
        path_to_save = os.path.join(path_to_test_data, dirs[i])

        #creates dir
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        files = get_files_from_folder(path_to_original)
        # moves data
        for j in range(int(test_counter[i])):
            dst = os.path.join(path_to_save, files[j])
            src = os.path.join(path_to_original, files[j])
            shutil.move(src, dst)

# How to use above Split_dataset function to split dataset into train and test set
"""
if __name__ == "__main__":
  args = parse_args()
  split_dataset(args.data_path, args.test_data_path_to_save, float(args.train_ratio))
"""
# with the given command
# python3 main.py ----data_path=/path1 --test_data_path_to_save=/path2 --train_ratio=0.7