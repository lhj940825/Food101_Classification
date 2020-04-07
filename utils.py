'''
 * User: Hojun Lim
 * Date: 2020-04-01
'''

import torch
import shutil
import os
import numpy as np
import argparse
import yaml


def is_gpu_available():
    """
    check whether the training goes with GPU or not
    :return: true or false
    """
    return torch.cuda.is_available()


def parse_yml(file_name):
    with open(file_name) as f:

        docs = yaml.load_all(f, Loader=yaml.FullLoader)

        for doc in docs:
            for k, v in doc.items():
                print(k, v)
                for v_s in v:
                    print(v_s.get('123d'))

    # TODO Add codes for parsing hyperparameters and directory paths, return those
    return


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


def split_dataset_into_3(path_to_dataset, train_ratio, valid_ratio):
    """
    split the dataset in the given path into three subsets(test,validation,train)

    :param path_to_dataset:
    :param train_ratio:
    :param valid_ratio:
    :return:
    """
    _, sub_dirs, _ = next(iter(os.walk(path_to_dataset)))  # retrieve name of subdirectories
    sub_dir_item_cnt = [0 for i in range(len(sub_dirs))]  # list for counting items in each sub directory(class)

    # directories where the splitted dataset will lie
    dir_train = os.path.join(os.path.dirname(path_to_dataset), 'train')
    dir_valid = os.path.join(os.path.dirname(path_to_dataset), 'validation')
    dir_test = os.path.join(os.path.dirname(path_to_dataset), 'test')

    for i, sub_dir in enumerate(sub_dirs):

        dir_train_dst = os.path.join(dir_train, sub_dir)  # directory for destination of train dataset
        dir_valid_dst = os.path.join(dir_valid, sub_dir)  # directory for destination of validation dataset
        dir_test_dst = os.path.join(dir_test, sub_dir)  # directory for destination of test dataset

        # variables to save the sub directory name(class name) and to count the images of each sub directory(class)
        class_name = sub_dir
        sub_dir = os.path.join(path_to_dataset, sub_dir)
        sub_dir_item_cnt[i] = len(os.listdir(sub_dir))

        items = os.listdir(sub_dir)

        # transfer data to trainset
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio)):
            if not os.path.exists(dir_train_dst):
                os.makedirs(dir_train_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_train_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to validation
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio) + 1,
                              round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio))):
            if not os.path.exists(dir_valid_dst):
                os.makedirs(dir_valid_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_valid_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to testset
        for item_idx in range(round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio)) + 1, sub_dir_item_cnt[i]):
            if not os.path.exists(dir_test_dst):
                os.makedirs(dir_test_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_test_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

    return


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

        # creates dir
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        files = get_files_from_folder(path_to_original)
        # moves data
        for j in range(int(test_counter[i])):
            dst = os.path.join(path_to_save, files[j])
            src = os.path.join(path_to_original, files[j])
            shutil.move(src, dst)


def cal_accuracy_by_class(model_predictions: torch.Tensor, labels: torch.Tensor, num_class):
    batch_size = len(labels)  # len(labels) = batch size
    c = model_predictions.eq(labels)
    class_correct = [0 for i in range(num_class)]
    class_total = [0 for i in range(num_class)]

    for i in range(batch_size):
        class_correct[labels[i]] += c[i] # c[i] = True or False
        class_total[labels[i]] += 1

    class_accuracies  = [100*x/y if y != 0 else 0 for x, y in zip(class_correct, class_total)]
    for i in range(num_class):
        print('Accuracy of class{} : {:3f} %'.format(i,class_accuracies[i]))


    return class_accuracies


# How to use above Split_dataset function to split dataset into train and test set
"""
if __name__ == "__main__":
  args = parse_args()
  split_dataset(args.data_path, args.test_data_path_to_save, float(args.train_ratio))
"""
# with the given command
# python3 main.py ----data_path=/path1 --test_data_path_to_save=/path2 --train_ratio=0.7


# How to use above split-dataset_into_3 function
if __name__ == "__main__":
    dir_data = os.path.join(os.getcwd(), 'dataset\\class10\\images')  # give the root directory of where dataset lies
    split_dataset_into_3(path_to_dataset=dir_data, train_ratio=0.7, valid_ratio=0.1)

#
