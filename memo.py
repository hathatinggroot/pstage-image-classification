import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import model.model as M

import argparse

def test():
    print('test....')



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='hello')
    
    args.add_argument('-c', '--config', type=str, required=True,
                    help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args = args.parse_args()