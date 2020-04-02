'''
 * User: Hojun Lim
 * Date: 2020-04-01
'''


import torch

def main():


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('123')


if __name__ == "__main__":
    main()
