import logging
import os
import pickle
import lmdb
import numpy as np
import pandas as pd
import six
import tqdm
from scipy.special import factorial
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

import utils
import utils.config as cfg
from .transforms import get_img_trans


def count_pairwise(count_array: np.ndarray, num_pairwise: int):
    """
    Get number of pair according to num_pairwise in count_array input
    
    params:
        count_array: np.ndarray (N,)
        num_pairwise: int
    """
    clear_count_array = count_array[count_array >= num_pairwise]
    count_pairwise_array = factorial(clear_count_array)
    return int(count_pairwise_array.sum())


def open_lmdb(path):
    return lmdb.open(
        path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False
    )


def load_semantic_data(semantic_fn):
    """Load semantic data."""
    data_fn = os.path.join(semantic_fn)
    with open(data_fn, "rb") as f:
        s2v = pickle.load(f)
    return s2v


class Datum(object):
    """
    Abstract class for Fashion Dataset.
    """

    def __init__(self,
                use_semantic=False,
                semantic=None,
                use_visual=False,
                image_dir="",
                lmdb_env=None,
                transforms=None,):
        self.cate_dict = cfg.CateIdx
        self.cate_name = cfg.CateName
        
        self.use_semantic = use_semantic
        self.semantic = semantic
        self.use_visual = use_visual
        self.image_dir = image_dir
        self.lmdb_env = lmdb_env
        self.transforms = transforms

    def load_image(self, id_name):
        """
        PIL loader for loading image.
        
        Return
        ------
        img: The image of idx name in image directory, type of PIL.Image.
        """
        img_name = f"{id_name}.jpg"
        if self.lmdb_env:
            # Read with lmdb format
            with self.lmdb_env.begin(write=False) as txn:
                imgbuf = txn.get(img_name.encode())
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        else:
            # Read from raw image
            path = os.path.join(self.image_dir, img_name)
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
        return img

    def visual_data(self, indices):
        """Load image data of the outfit."""
        images = []
        for id_name in indices:
            if id_name == -1:
                ##TODO: Dynamic this later
                img = torch.zeros((3, 300, 300), dtype=torch.float32)
            else:
                img = self.load_image(id_name)
                if self.transforms:
                    img = self.transforms(img)
            images.append(img)
        return images

    def get(self, tpl):
        """Convert a tuple to torch.FloatTensor"""
        if self.use_semantic and self.use_visual:
            tpl_s = self.semantic_data(tpl)
            tpl_v = self.visual_data(tpl)
            return tpl_v, tpl_s
        if self.use_visual:
            return self.visual_data(tpl)
        if self.use_semantic:
            return self.semantic_data(tpl)
        return tpl


##TODO: Merge with FashionDataset
class FashionExtractionDataset(Dataset):
    def __init__(self, param, transforms=None, cate_selection="all", logger=None):
        self.param = param
        self.logger = logger

        self.df = pd.read_csv(self.param.data_csv)

        # After processing
        if cate_selection == "all":
            cate_selection = list(self.df.columns)
        else:
            cate_selection = cate_selection + ["compatible",]

        ##TODO: Simplify this later
        self.cate_idxs = [cfg.CateIdx[col] for col in cate_selection[:-1]]
        self.cate_idxs_to_tensor_idxs = {cate_idx: tensor_idx for cate_idx, tensor_idx in zip(self.cate_idxs, range(len(self.cate_idxs)))}
        self.tensor_idxs_to_cate_idxs = {v: k for k, v in self.cate_idxs_to_tensor_idxs.items()}
        
        self.df = self.get_new_data_with_new_cate_selection(self.df, cate_selection)

        self.df = self.df.drop("compatible", axis=1)

        if param.use_semantic:
            ##TODO: Code this later
            semantic = load_semantic_data(param.semantic_fn)
        else:
            semantic = None

        ##TODO: Careful with lmdb
        lmdb_env = open_lmdb(param.lmdb_dir) if param.use_lmdb else None
        self.datum = Datum(
            use_semantic=param.use_semantic,
            semantic=semantic,
            use_visual=param.use_visual,
            image_dir=param.image_dir,
            lmdb_env=lmdb_env,
            transforms=transforms
        )

    def get_new_data_with_new_cate_selection(self, df, cate_selection):
        df = df.copy()
        df = df[cate_selection]
        df_count = (df.to_numpy()[..., :-1] != -1).astype(int).sum(axis=-1)
        return df[df_count > 1]

    def get_tuple(self, idx):
        raw_tuple = self.df.iloc[idx]
        outfit_tuple = raw_tuple[raw_tuple != -1]
        outfit_idxs = [cfg.CateIdx[col] for col in outfit_tuple.index.to_list()]
        return outfit_idxs, outfit_tuple.values.tolist()

    def __getitem__(self, index):
        """Get one tuple of examples by index."""
        idxs, tpl = self.get_tuple(index)
        return idxs, tpl, self.datum.get(tpl)

    def __len__(self):
        """Return the size of dataset."""
        return len(self.df)


class FashionDataset(Dataset):
    def __init__(self, param, transforms=None, cate_selection="all", logger=None):
        self.param = param
        self.logger = logger

        self.df = pd.read_csv(self.param.data_csv)
        num_pairwise_list = param.num_pairwise
        self.logger.info("Dataframe processing...")
        # Before processing
        num_row_before = len(self.df)
        pairwise_count_before_list = self.get_pair_list(num_pairwise_list, self.df)
        self.logger.info(f"+ Before: Num row: {utils.colour(num_row_before)} - " + \
                        " - ".join([f"pairwise {num_pairwise}: {utils.colour(pairwise_count_before)}" for \
                                    num_pairwise, pairwise_count_before in zip(num_pairwise_list, pairwise_count_before_list)]))
        self.logger.info("")

        # After processing
        if cate_selection == "all":
            cate_selection = list(self.df.columns)
        else:
            cate_selection = cate_selection + ["compatible",]

        ##TODO: Simplify this later
        self.cate_idxs = [cfg.CateIdx[col] for col in cate_selection[:-1]]
        self.cate_idxs_to_tensor_idxs = {cate_idx: tensor_idx for cate_idx, tensor_idx in zip(self.cate_idxs, range(len(self.cate_idxs)))}
        self.tensor_idxs_to_cate_idxs = {v: k for k, v in self.cate_idxs_to_tensor_idxs.items()}
        
        self.df = self.get_new_data_with_new_cate_selection(self.df, cate_selection)
        num_row_after = len(self.df)
        pairwise_count_after_list = self.get_pair_list(num_pairwise_list, self.df)
        self.logger.info(f"+ After: Num row: {utils.colour(num_row_after)} - " + \
                        " - ".join([f"pairwise {num_pairwise}: {utils.colour(pairwise_count_after)}" for \
                                    num_pairwise, pairwise_count_after in zip(num_pairwise_list, pairwise_count_after_list)]))
        self.logger.info("")

        self.posi_df_ori = self.df[self.df.compatible == 1].reset_index(drop=True).drop("compatible", axis=1)
        self.nega_df_ori = self.df[self.df.compatible == 0].reset_index(drop=True).drop("compatible", axis=1)
        
        assert len(self.posi_df_ori) + len(self.nega_df_ori) == len(self.df)

        self.posi_df = self.posi_df_ori.copy()

        if param.use_semantic:
            ##TODO: Code this later
            semantic = load_semantic_data(param.semantic_fn)
        else:
            semantic = None

        ##TODO: Careful with lmdb
        lmdb_env = open_lmdb(param.lmdb_dir) if param.use_lmdb else None
        self.datum = Datum(
            use_semantic=param.use_semantic,
            semantic=semantic,
            use_visual=param.use_visual,
            image_dir=param.image_dir,
            lmdb_env=lmdb_env,
            transforms=transforms
        )
        self.using_max_num_pairwise = param.using_max_num_pairwise

        # probability for hard negative samples
        self.hard_ratio = 0.8
        # the ratio between negative outfits and positive outfits
        self.ratio = self.ratio_fix = len(self.nega_df_ori) / len(self.posi_df_ori)
        self.set_data_mode(param.data_mode)
        self.set_nega_mode(param.nega_mode)

    ##TODO: Modify this, do we need this
    def set_nega_mode(self, mode):
        """Set negative outfits mode."""
        assert mode in [
            "ShuffleDatabase",
            "RandomOnline",
            "RandomFix",
            "HardOnline",
            "HardFix",
        ], "Unknown negative mode."
        if self.param.data_mode == "PosiOnly":
            self.logger.warning(
                f"Current data-mode is {utils.colour(self.param.data_mode, 'Red')}." \
                "The negative mode will be ignored!",
            )
        else:
            self.logger.info(f"Set negative mode to {utils.colour(mode)}")
            self.param.nega_mode = mode
            self.make_nega()

    def _shuffle_nega(self,):
        return self.nega_df_ori.sample(frac=1).reset_index(drop=True)

    def make_nega(self, ratio=1):
        """Make negative outfits according to its mode and ratio."""
        self.logger.info("Make negative outfit for mode %s" % self.param.nega_mode)
        if self.param.nega_mode == "ShuffleDatabase":
            self.nega_df = self._shuffle_nega()
            self.logger.info("Shuffle negative database")
        elif self.param.nega_mode == "RandomOnline":
            ##TODO: Random the negative dataframe from positive one
            raise
        else:
            raise ##TODO:
        self.logger.info("Done making negative outfits!")

    ##TODO: Modify this func
    def set_data_mode(self, mode):
        """Set data mode."""
        assert mode in [
            "TupleOnly",
            "PosiOnly",
            "NegaOnly",
            "PairWise",
            "TripleWise"
        ], (f"Unknown data mode: {mode}")
        self.logger.info(f"Set data mode to {utils.colour(mode)}")
        self.param.data_mode = mode

    ##TODO: What is it?
    def set_prob_hard(self, p):
        """Set the proportion for hard negative examples."""
        if self.param.data_mode == "PosiOnly":
            self.logger.warning(
                "Current data-mode is %s. " "The proportion will be ignored!",
                utils.colour(self.param.data_mode, "Red"),
            )
        elif self.param.nega_mode != "HardOnline":
            self.logger.warning(
                "Current negative-sample mode is %s. "
                "The proportion will be ignored!",
                utils.colour(self.param.nega_mode, "Red"),
            )
        else:
            self.phard = p
            self.logger.info(
                "Set the proportion of hard negative outfits to %s",
                utils.colour("%.3f" % p),
            )

    def get_new_data_with_new_cate_selection(self, df, cate_selection):
        df = df.copy()
        df = df[cate_selection]
        df_count = (df.to_numpy()[..., :-1] != -1).astype(int).sum(axis=-1)
        return df[df_count > 1]

    def get_pair_list(self, num_pairwise_list, df):
        # for i, row in df.iterrows()
        df_array = df.to_numpy()[..., :-1]  # Eliminate compatible
        df_count = (df_array != -1).astype(int).sum(axis=-1)
        
        pairwise_count_list = []
        for num_pairwise in num_pairwise_list:
            pairwise_count_list.append(count_pairwise(df_count, num_pairwise))
        return pairwise_count_list

    ##TODO:
    # def get_tuple(self, df, idx):
    #     outfit_idxs_out = torch.zeros(len(df.columns))
    #     raw_tuple = df.iloc[idx]
    #     outfit_tuple = raw_tuple[raw_tuple != -1]
    #     outfit_idxs = [cfg.CateIdx[col] for col in outfit_tuple.index.to_list()]
    #     outfit_tensor_idxs = [self.cate_idxs_to_tensor_idxs[outfit_idx] for outfit_idx in outfit_idxs]
    #     outfit_idxs_out[outfit_tensor_idxs] = 1
    #     return outfit_idxs, raw_tuple.values.tolist()

    def get_tuple(self, df, idx):
        raw_tuple = df.iloc[idx]
        outfit_tuple = raw_tuple[raw_tuple != -1]
        outfit_idxs = [cfg.CateIdx[col] for col in outfit_tuple.index.to_list()]
        return outfit_idxs, outfit_tuple.values.tolist()

    def _PairWise(self, index):
        """Get a pair of outfits."""
        ##TODO: Modify index for tuple selection for posi and nega (maybe shuffle the df each epoch)
        posi_idxs, posi_tpl = self.get_tuple(self.posi_df, int(index // self.ratio))
        nega_idxs, nega_tpl = self.get_tuple(self.nega_df, index)
        return ((posi_idxs, self.datum.get(posi_tpl)), (nega_idxs, self.datum.get(nega_tpl)))

    def __getitem__(self, index):
        """Get one tuple of examples by index."""
        return dict(
            PairWise=self._PairWise,
        )[self.param.data_mode](index)

    def __len__(self):
        """Return the size of dataset."""
        return dict(
            PairWise=int(self.ratio * self.num_posi)
        )[self.param.data_mode]

    @property
    def num_posi(self):
        """Number of positive outfit."""
        return len(self.posi_df)

    @property
    def num_nega(self):
        """Number of negative outfit."""
        return len(self.nega_df)


class FashionLoader(object):
    """Class for Fashion data loader"""

    def __init__(self, param, logger):
        self.logger = logger

        self.cate_selection = param.cate_selection
        self.cate_not_selection = [cate for cate in cfg.CateName if cate not in param.cate_selection]

        self.logger.info(
            f"Loading data ({utils.colour(param.data_set)}) in phase ({utils.colour(param.phase)})"
        )
        self.logger.info(
            f"- Selected apparel: " + ", ".join([utils.colour(cate) for cate in self.cate_selection])
        )
        self.logger.info(
            f"- Not selected apparel: " + ", ".join([utils.colour(cate, "Red") for cate in self.cate_not_selection])
        )
        self.logger.info(
            f"- Data loader configuration: batch size ({utils.colour(param.batch_size)}), number of workers ({utils.colour(param.num_workers)})"
        )
        transforms = get_img_trans(param.phase, param.image_size)
        self.dataset = FashionDataset(param, transforms, self.cate_selection.copy(), logger)
        
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=param.batch_size,
            num_workers=param.num_workers,
            shuffle=param.shuffle,
            pin_memory=True,
            collate_fn=outfit_fashion_collate,
        )

    def __len__(self):
        """Return number of batches."""
        return len(self.loader)

    @property
    def num_batch(self):
        """Get number of batches."""
        return len(self.loader)

    @property
    def num_sample(self):
        """Get number of samples."""
        return len(self.dataset)

    def make_nega(self, ratio=1):
        """Prepare negative outfits."""
        self.dataset.make_nega(ratio)
        return self

    def set_nega_mode(self, mode):
        """Set the mode for generating negative outfits."""
        self.dataset.set_nega_mode(mode)
        return self

    def set_data_mode(self, mode):
        """Set the mode for data set."""
        self.dataset.set_data_mode(mode)
        return self

    def set_prob_hard(self, p):
        """Set the probability of negative outfits."""
        self.dataset.set_prob_hard(p)
        return self

    def __iter__(self):
        """Return generator."""
        for data in self.loader:
            yield data


def outfit_fashion_collate(batch):
    """Custom collate function for dealing with batch of fashion dataset
    Each sample will has following output from dataset:
        ((`posi_idxs`, `posi_imgs`), (`nega_idxs`, `nega_imgs`))
        ----------
        - Examples
            `posi_idxs`: [i1, i2, i3]
            `posi_imgs`: [(3, 300, 300), (3, 300, 300), (3, 300, 300)]
            `nega_idxs`: [i1, i2]
            `nega_imgs`: [(3, 300, 300), (3, 300, 300)]
        ----------
        The number of apparels in each list is different between different sample
        We need concatenate them wisely

    Outputs:
        ##TODO: Describe later
    --------
    """
    posi_mask, posi_idxs_out, posi_imgs_out, nega_mask, nega_idxs_out, nega_imgs_out = \
            [], [], [], [], [], []
    
    for i, sample in enumerate(batch):
        (posi_idxs, posi_imgs), (nega_idxs, nega_imgs) = sample

        posi_mask.extend([i]*len(posi_idxs))
        posi_idxs_out.extend(posi_idxs)
        posi_imgs_out.extend(posi_imgs)

        nega_mask.extend([i]*len(nega_idxs))
        nega_idxs_out.extend(nega_idxs)
        nega_imgs_out.extend(nega_imgs)
    
    return torch.Tensor(posi_mask).to(torch.long), torch.Tensor(posi_idxs_out).to(torch.long), torch.stack(posi_imgs_out, 0), \
            torch.Tensor(nega_mask).to(torch.long), torch.Tensor(nega_idxs_out).to(torch.long), torch.stack(nega_imgs_out, 0)


# --------------------------
# Loader and Dataset Factory
# --------------------------


def get_dataloader(param, logger):
    name = param.__class__.__name__
    if name == "DataParam":
        return FashionLoader(param, logger)
    return None