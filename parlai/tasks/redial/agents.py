import copy
import re
import csv
import json
import os
import pickle as pkl

import requests

import parlai.core.agents as core_agents
from parlai.core.teachers import DialogTeacher

from .build import build


def _path(opt):
    # ensure data is built
    build(opt)

    # set up paths to data (specific to each dataset)
    dt = opt["datatype"].split(":")[0]
    return (
        os.path.join(opt["datapath"], "redial", f"{dt}_data.jsonl"),
        os.path.join(opt["datapath"], "redial", "movies_with_mentions.csv"),
        os.path.join(opt["datapath"], "redial", "id2entity.pkl"),
    )


# Set up a local dbpedia-spotlight docker https://github.com/dbpedia-spotlight/spotlight-docker
DBPEDIA_SPOTLIGHT_ADDR = " http://0.0.0.0:2222/rest/annotate"
SPOTLIGHT_CONFIDENCE = 0.3


def _id2dbpedia(movie_id):
    pass


def _text2entities(text):
    headers = {"accept": "application/json"}
    params = {"text": text, "confidence": SPOTLIGHT_CONFIDENCE}

    response = requests.get(
        DBPEDIA_SPOTLIGHT_ADDR, headers=headers, params=params
    ).json()
    return (
        [f"<{x['@URI']}>" for x in response["Resources"]]
        if "Resources" in response
        else []
    )


class RedialTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.datatype = opt["datatype"].split(":")[0]

        # store identifier for the teacher in the dialog
        self.id = "redial"

        # store paths to images and labels
        opt["datafile"], movies_with_mentions_path, id2entity_path = _path(opt)

        with open(movies_with_mentions_path, "r") as f:
            reader = csv.reader(f)
            self.id2name, self.id2idx = {}, {}
            for idx, row in enumerate(reader):
                if row[0] == "movieId":
                    continue
                self.id2name["@" + row[0]] = row[1]
                self.id2idx["@" + row[0]] = idx - 1

        self.id2entity = pkl.load(open(id2entity_path, "rb"))

        super().__init__(opt, shared)

    def _convert_ids_to_indices(self, text, questions):
        """@movieID -> @movieIdx#like"""
        pattern = re.compile("@\d+")

        def convert(match):
            movieId = match.group(0)
            if movieId[1:] in questions and questions[movieId[1:]]["liked"] in [0, 1]:
                return (
                    "@" + str(self.id2idx[movieId]) + "#" + str(questions[movieId[1:]]["liked"])
                )
            else:
                return "@" + str(self.id2idx[movieId])

        return re.sub(pattern, convert, text)

    def setup_data(self, path):
        self.instances = []
        with open(path) as json_file:
            for line in json_file.readlines():
                self.instances.append(json.loads(line))

        # define iterator over all queries
        for instance in self.instances:
            initiator_id = instance["initiatorWorkerId"]
            respondent_id = instance["respondentWorkerId"]
            messages = instance["messages"]
            message_idx = 0
            new_episode = True

            while message_idx < len(messages):
                source_text = ""
                target_text = ""
                while (
                    message_idx < len(messages)
                    and messages[message_idx]["senderWorkerId"] == initiator_id
                ):
                    source_text += messages[message_idx]["text"] + " <SEP> "
                    message_idx += 1
                while (
                    message_idx < len(messages)
                    and messages[message_idx]["senderWorkerId"] == respondent_id
                ):
                    target_text += messages[message_idx]["text"] + " <SEP> "
                    message_idx += 1
                if source_text != "" and target_text != "":
                    # remove the trailing <SEP>
                    source_text, target_text = source_text[:-6], target_text[:-6]
                    # convert movieId to index [0..n_movies-1]
                    source_text = self._convert_ids_to_indices(
                        source_text, instance["initiatorQuestions"]
                    )
                    target_text = self._convert_ids_to_indices(
                        target_text, instance["initiatorQuestions"]
                    )
                    yield (source_text, [target_text], None, None, None), new_episode
                    new_episode = False


class DefaultTeacher(RedialTeacher):
    pass
