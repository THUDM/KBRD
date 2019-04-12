import json
import os
import random

import parlai.core.build_data as build_data


def split_data(redial_path):
    # Copied from https://github.com/RaymondLi0/conversational-recommendations/blob/master/scripts/split-redial.py
    data = []
    for line in open(os.path.join(redial_path, "train_data.jsonl")):
        data.append(json.loads(line))
    random.shuffle(data)
    n_data = len(data)
    split_data = [data[: int(0.9 * n_data)], data[int(0.9 * n_data) :]]

    with open(os.path.join(redial_path, "train_data.jsonl"), "w") as outfile:
        for example in split_data[0]:
            json.dump(example, outfile)
            outfile.write("\n")
    with open(os.path.join(redial_path, "valid_data.jsonl"), "w") as outfile:
        for example in split_data[1]:
            json.dump(example, outfile)
            outfile.write("\n")


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt["datapath"], "redial")
    # define version if any
    version = None

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print("[building data: " + dpath + "]")

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # download the data.
        fname = "redial_dataset.zip"
        url = "https://github.com/ReDialData/website/raw/data/" + fname  # dataset URL
        build_data.download(url, dpath, fname)

        # uncompress it
        build_data.untar(dpath, fname)

        split_data(dpath)

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
