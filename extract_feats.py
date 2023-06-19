import argparse
import logging
import os
import yaml
import pickle
import numpy as np
import warnings
from tqdm import tqdm
from time import time
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

import torch

import utils
import utils.config as cfg
from utils.param import FashionExtractParam
from utils.logger import Logger, config_log
from dataset.fashionset import FashionExtractionDataset
from dataset.transforms import get_img_trans
from model import FashionNet


def get_logger(env, config):
    if env == "local":
        ##TODO: Modify this logger name
        logfile = config_log(stream_level=config.log_level, log_file=config.log_file)
        logger = logging.getLogger("polyvore")
        logger.info("Logging to file %s", logfile)
    elif env == "colab":
        logger = Logger(config)  # Normal logger
        logger.info(f"Logging to file {logger.logfile}")

    return logger


def get_dataset(data_param, logger):
    transforms = get_img_trans("val", data_param.image_size)
    dataset = FashionExtractionDataset(data_param, transforms, data_param.cate_selection, logger)

    return dataset


def get_net(config, logger):
    """Get network."""
    # Get net param
    net_param = config.net_param
    logger.info(f"Initializing {utils.colour(config.net_param.name)}")
    logger.info(net_param)
    
    assert config.load_trained is not None

    # Dimension of latent codes
    net = FashionNet(net_param, logger, 
                    config.train_data_param.cate_selection)
    # Load model from pre-trained file
    num_devices = torch.cuda.device_count()
    map_location = {"cuda:{}".format(i): "cpu" for i in range(num_devices)}
    logger.info(f"Loading pre-trained model from {config.load_trained}")
    state_dict = torch.load(config.load_trained, map_location=map_location)
    # load pre-trained model
    net.load_state_dict(state_dict)
    logger.info(f"Copying net to GPU-{config.gpus[0]}")
    net.cuda(device=config.gpus[0])
    net.eval()  # Unable training
    return net


def main(config, logger):
    """Feature extraction task"""
    ##TODO: Dynamic for train, val, test options
    # Get train dataset
    train_param = config.train_data_param or config.data_param
    logger.info(f"Dataset for train: \n{train_param}")
    trainset = get_dataset(train_param, logger)
    # Get valid dataset
    val_param = config.test_data_param or config.data_param
    logger.info(f"Data set for val: \n{val_param}")
    valset = get_dataset(val_param, logger)

    phases = ["train", "val"]  ##TODO: test??
    datasets = [trainset, valset]

    cat2idx = cfg.CateIdx
    idx2cat = {v:k for k, v in cat2idx.items()}
    cate_idxs = trainset.cate_idxs

    feats_latent_visual_dict = {}
    # feats_latent_semantic_dict = {}  ##TODO: Later
    feats_binary_visual_dict = {}
    # feats_binary_semantic_dict = {}
    for phase in phases:
        feats_latent_visual_dict[phase] = {cate: {} for cate in train_param.cate_selection}
        feats_binary_visual_dict[phase] = {cate: {} for cate in train_param.cate_selection}

    # Get net
    net = get_net(config, logger)
    device = config.gpus[0]
    data_times = []
    model_run_times = []

    # Get features embedding
    lastest_time = time()
    for phase, dataset in zip(phases, datasets):
        for data_input in tqdm(dataset, desc="Trainset extraction:", total=len(dataset)):
            outfit_idxs, tpl_names, inputs = data_input
            inputs = torch.stack(inputs, 0)
            inputs = utils.to_device(inputs, device)
            data_time = time() - lastest_time
            lcis_v, lcis_s, bcis_v, bcis_s = net.extract_features(inputs)

            lcis_v = [lci_v for lci_v in lcis_v]
            lcis_s = [None for _ in range(len(lcis_v))]  ##TODO: Modify for semantic later
            bcis_v = [bci_v for bci_v in bcis_v]
            bcis_s = [None for _ in range(len(bcis_v))]

            model_run_time = time() - lastest_time
            
            for outfit_idx, sample_name, lci_v, lci_s, bci_v, bci_s in \
                    zip(outfit_idxs, tpl_names, lcis_v, lcis_s, bcis_v, bcis_s):
                cate_name = idx2cat[outfit_idx]
                feats_latent_visual_dict[phase][cate_name][str(sample_name)] = lci_v
                feats_binary_visual_dict[phase][cate_name][str(sample_name)] = bci_v

            data_times.append(data_time)
            model_run_times.append(model_run_time)
        
        feats_latent_visual_file = os.path.join(config.feature_folder, f"{phase}_latent_visual.pkl")
        feats_binary_visual_file = os.path.join(config.feature_folder, f"{phase}_binary_visual.pkl")

        with open(feats_latent_visual_file, "wb") as handle:
            pickle.dump(feats_latent_visual_dict[phase], handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(feats_binary_visual_file, "wb") as handle:
            pickle.dump(feats_binary_visual_dict[phase], handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"--Total data time: {np.sum(data_times), :.3f}s")
    print(f"--Average data time: {np.mean(data_times), :.3f}s")
    print(f"--Total model run time: {np.sum(data_times), :.3f}s")
    print(f"--Average model run time: {np.sum(data_times), :.3f}s")


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
    config = FashionExtractParam(**kwargs)

    os.makedirs(config.feature_folder, exist_ok=True)

    logger = get_logger(args.env, config)
    logger.info(f"Fashion param : {config}")

    main(config, logger)