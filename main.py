import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from config import build_arg_parser, config_from_args
from trainer import train_single_env


def main():
    parser = build_arg_parser(vector=False)
    args = parser.parse_args()
    config = config_from_args(args, vector=False)
    train_single_env(config)


if __name__ == "__main__":
    main()
