import copy
import re
import csv
import json
import os
import pickle as pkl

import requests

import parlai.core.agents as core_agents
from parlai.core.teachers import DialogTeacher
from parlai.core.dict import DictionaryAgent

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
        os.path.join(opt["datapath"], "redial", "entity_dict.pkl"),
        os.path.join(opt["datapath"], "redial", "text_dict.pkl"),
        os.path.join(opt["datapath"], "redial", "entity2entityId.pkl"),
        os.path.join(opt["datapath"], "redial", "relation2relationId.pkl"),
    )


def _id2dbpedia(movie_id):
    pass


def _text2entities(text, text_dict):
    return text_dict[text]


class RedialTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.datatype = opt["datatype"].split(":")[0]

        # store identifier for the teacher in the dialog
        self.id = "redial"

        # store paths to images and labels
        opt[
            "datafile"
        ], movies_with_mentions_path, id2entity_path, entity_dict_path, text_dict_path, entity2entityId_path, relation2relationId_path = _path(
            opt
        )

        if not shared:
            self.entity2entityId = pkl.load(open(entity2entityId_path, "rb"))
            self.relation2relationId = pkl.load(open(relation2relationId_path, "rb"))
            self.id2entity = pkl.load(open(id2entity_path, "rb"))
            self.text_dict = pkl.load(open(text_dict_path, "rb"))
        else:
            self.entity2entityId = shared["entity2entityId"]
            self.relation2relationId = shared["relation2relationId"]
            self.id2entity = shared["id2entity"]
            self.text_dict = shared["text_dict"]

        super().__init__(opt, shared)

    def share(self):
        """Share internal states."""
        shared = super().share()
        shared["entity2entityId"] = self.entity2entityId
        shared["relation2relationId"] = self.relation2relationId
        shared["id2entity"] = self.id2entity
        shared["text_dict"] = self.text_dict
        return shared

    def _convert_ids_to_indices(self, text, questions):
        """@movieID -> @movieIdx"""
        pattern = re.compile("@\d+")
        movieId_list = []

        def convert(match):
            movieId = match.group(0)
            try:
                entity = self.id2entity[int(movieId[1:])]
                if entity is not None:
                    movieId_list.append(str(self.entity2entityId[entity]))
                else:
                    movieId_list.append(str(self.entity2entityId[int(movieId[1:])]))
                return DictionaryAgent.default_unk
            except Exception:
                return ""

        return re.sub(pattern, convert, text), movieId_list

    def _get_entities(self, text):
        """text -> [#entity1, #entity2]"""
        entities = _text2entities(text, self.text_dict)
        entities = [str(self.entity2entityId[x]) for x in entities if x in self.entity2entityId]
        return entities

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

            previously_mentioned_movies_list = []
            mentioned_entities = []
            turn = 0
            while message_idx < len(messages):
                source_text = []
                target_text = []
                while (
                    message_idx < len(messages)
                    and messages[message_idx]["senderWorkerId"] == initiator_id
                ):
                    source_text.append(messages[message_idx]["text"])
                    message_idx += 1
                while (
                    message_idx < len(messages)
                    and messages[message_idx]["senderWorkerId"] == respondent_id
                ):
                    target_text.append(messages[message_idx]["text"])
                    message_idx += 1
                source_text = [text for text in source_text if text != ""]
                target_text = [text for text in target_text if text != ""]
                if source_text != [] or target_text != []:
                    for src in source_text:
                        mentioned_entities += self._get_entities(src)
                    target_mentioned_entities = []
                    for tgt in target_text:
                        target_mentioned_entities += self._get_entities(tgt)
                    source_text = '\n'.join(source_text)
                    target_text = '\n'.join(target_text)
                    source_text, source_movie_list = self._convert_ids_to_indices(
                        source_text, instance["initiatorQuestions"]
                    )
                    target_text, target_movie_list = self._convert_ids_to_indices(
                        target_text, instance["initiatorQuestions"]
                    )
                    turn += 1
                    if message_idx == len(messages) and target_text == "":
                        break
                    yield (source_text, [target_text], None, [str(turn), ' '.join(previously_mentioned_movies_list + source_movie_list), ' '.join(target_movie_list), ' '.join(mentioned_entities), target_text], None), new_episode
                    new_episode = False
                    previously_mentioned_movies_list += source_movie_list + target_movie_list
                    mentioned_entities += target_mentioned_entities


class DefaultTeacher(RedialTeacher):
    pass
