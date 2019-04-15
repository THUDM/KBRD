import csv
import json
import os
import pickle as pkl
import random
import re
from collections import defaultdict

import parlai.core.build_data as build_data


def _split_data(redial_path):
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


def _entity2movie(entity, abstract=""):
    # strip url
    x = entity[::-1].find("/")
    movie = entity[-x:-1]
    movie = movie.replace("_", " ")

    # extract year
    pattern = re.compile(r"\d{4}")
    match = re.findall(pattern, movie)
    year = match[0] if match else None
    # if not find in entity title, find in abstract
    if year is None:
        pattern = re.compile(r"\d{4}")
        match = re.findall(pattern, abstract)
        if match and 1900 < int(match[0]) and int(match[0]) < 2020:
            year = match[0]

    # recognize (20xx film) or (film) to help disambiguation
    pattern = re.compile(r"\(.*film.*\)")
    match = re.findall(pattern, movie)
    definitely_is_a_film = match != []

    # remove parentheses
    while True:
        pattern = re.compile(r"(.+)( \(.*\))")
        match = re.search(pattern, movie)
        if match:
            movie = match.group(1)
        else:
            break
    movie = movie.strip()

    return movie, year, definitely_is_a_film


DBPEDIA_ABSTRACT_PATH = "dbpedia/short_abstracts_en.ttl"


def _build_dbpedia(dbpedia_path):
    movie2entity = {}
    movie2years = defaultdict(set)
    with open(dbpedia_path) as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            entity, line = line[: line.index(" ")], line[line.index(" ") + 1 :]
            _, line = line[: line.index(" ")], line[line.index(" ") + 1 :]
            abstract = line[:-4]
            movie, year, definitely_is_a_film = _entity2movie(entity, abstract)
            if (movie, year) not in movie2entity or definitely_is_a_film:
                movie2years[movie].add(year)
                movie2entity[(movie, year)] = entity
    return {"movie2years": movie2years, "movie2entity": movie2entity}


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

        _split_data(dpath)

        dbpedia = _build_dbpedia(DBPEDIA_ABSTRACT_PATH)
        movie2entity = dbpedia["movie2entity"]
        movie2years = dbpedia["movie2years"]

        movies_with_mentions_path = os.path.join(dpath, "movies_with_mentions.csv")
        with open(movies_with_mentions_path, "r") as f:
            reader = csv.reader(f)
            id2movie = {int(row[0]): row[1] for row in reader if row[0] != "movieId"}
        id2entity = {}
        for movie_id in id2movie:
            movie = id2movie[movie_id]
            pattern = re.compile(r"(.+)\((\d+)\)")
            match = re.search(pattern, movie)
            if match is not None:
                name, year = match.group(1).strip(), match.group(2)
            else:
                name, year = movie[1].strip(), None
            if year is not None:
                if (name, year) in movie2entity:
                    id2entity[movie_id] = movie2entity[(name, year)]
                else:
                    if len(movie2years) == 1:
                        id2entity[movie_id] = movie2entity[(name, movie2years[name][0])]
                    else:
                        id2entity[movie_id] = None

            else:
                id2entity[movie_id] = (
                    movie2entity[(name, year)] if (name, year) in movie2entity else None
                )

        pkl.dump(id2entity, open(os.path.join(dpath, "id2entity.pkl"), "wb"))
        pkl.dump(dbpedia, open(os.path.join(dpath, "dbpedia.pkl"), "wb"))

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
