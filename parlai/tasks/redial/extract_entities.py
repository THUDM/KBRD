import json
import re
import pickle as pkl
from collections import defaultdict

import requests
from tqdm import tqdm

# Set up a local dbpedia-spotlight docker https://github.com/dbpedia-spotlight/spotlight-docker
DBPEDIA_SPOTLIGHT_ADDR = " http://0.0.0.0:2222/rest/annotate"
SPOTLIGHT_CONFIDENCE = 0.1


def _id2dbpedia(movie_id):
    pass


def _text2entities(text):
    headers = {"accept": "application/json"}
    params = {"text": text, "confidence": SPOTLIGHT_CONFIDENCE}

    response = requests.get(DBPEDIA_SPOTLIGHT_ADDR, headers=headers, params=params)
    response = response.json()
    return (
        [f"<{x['@URI']}>" for x in response["Resources"]]
        if "Resources" in response
        else []
    )


def _tags(split, tags_dict, text_dict):
    path = f"data/redial/{split}_data.jsonl"
    instances = []
    with open(path) as json_file:
        for line in json_file.readlines():
            instances.append(json.loads(line))

    print(split, len(instances))

    num_tags = 0
    pattern = re.compile("@\d+")
    for instance in tqdm(instances):
        initiator_id = instance["initiatorWorkerId"]
        respondent_id = instance["respondentWorkerId"]
        messages = instance["messages"]
        for message in messages:
            if message["text"] != "":
                tags = _text2entities(message["text"])
                text_dict[message["text"]] = tags
                num_tags += len(tags)
                for tag in tags:
                    tags_dict[tag] += 1

tags_dict = defaultdict(int)
text_dict = {}
for split in ['train', 'valid', 'test']:
    _tags(split, tags_dict, text_dict)

print(len(tags_dict))

pkl.dump(text_dict, open('data/redial/text_dict_confidence_0.1.pkl', 'wb'))
pkl.dump(tags_dict, open('data/redial/entity_dict_confidence_0.1.pkl', 'wb'))
