"""Script for training."""
import argparse
import logging
import os
import pickle
import shutil
import textwrap
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import torch
import tqdm
import yaml
from torch.nn.parallel import data_parallel

from utils.param import FashionTrainParam
from utils.logger import Logger, config_log


def get_logger(env, config):
    if env == "local":
        ##TODO: Modify this logger name
        logfile = config_log(stream_level=config.log_level, log_file=config.log_file)
        logger = logging.getLogger("polyvore")
        logger.info("Logging to file %s", logfile)
    elif env == "colab":
        logger = Logger(config)  # Normal logger
        logger.info(f"Logging to file {logger.logfile}")

    logger.info(f"Fashion param : {config}")


def main(config, logger):
    """Training task"""
    # Get data for training
    train_param = config.train_data_param or config.data_param
    # logger.info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Hash for All Fashion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Hashing for All Fashion scripts"
    )
    parser.add_argument("--cfg", help="configuration file.")
    parser.add_argument("--env", default="local", choices=["local", "colab"], 
                    help="Using for logging option. Using logger if local, using normal print otherwise.")
    args = parser.parse_args()
    with open(args.cfg, "r") as f:
        kwargs = yaml.load(f, Loader=yaml.FullLoader)
    config = FashionTrainParam(**kwargs)
    config.add_timestamp()

    logger = get_logger(args.env, config)

    main(config, logger)