import argparse
import os
import lmdb
from tqdm import tqdm

import utils


def create_lmdb(dataset, image_dir):
    print(f"Creating LMDB to {dataset}")
    image_list = utils.check.list_files(image_dir)
    env = lmdb.open(dataset, map_size=2 ** 40)
    # open json file
    with env.begin(write=True) as txn:
        for image_name in tqdm(image_list):
            fn = os.path.join(image_dir, image_name)
            with open(fn, "rb") as f:
                img_data = f.read()
                txn.put(image_name.encode("ascii"), img_data)
    env.close()
    print("Converted dataset to LMDB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create LMDB")
    parser.add_argument(
        "--src",
        type=str,
        help="image folder for fashion dataset",
        required=True
    )
    parser.add_argument("--dst", type=str, help="folder to save lmdb", required=True)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    create_lmdb(args.dst, args.src)
    exit()