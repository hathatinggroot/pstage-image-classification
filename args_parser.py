
import argparse


class CustomArgs:
    def __init__(self, description='Train') -> None:
        args = argparse.ArgumentParser(description=f'[{description}]]')

        args.add_argument('-c', '--config', type=str, required=True,
                        help='config file path (default: None)')
        args.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
        args.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
        self.args = args.parse_args()