from parlai.core.teachers import DialogTeacher
import parlai.core.agents as core_agents
from .build import build

import copy
import os
import json
import csv


def _path(opt):
    # ensure data is built
    build(opt)

    # set up paths to data (specific to each dataset)
    dt = opt["datatype"].split(":")[0]
    return (
        os.path.join(opt["datapath"], "redial", f"{dt}_data.jsonl"),
        os.path.join(opt["datapath"], "redial", "movies_with_mentions.csv"),
    )


class RedialTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.datatype = opt["datatype"].split(":")[0]

        # store identifier for the teacher in the dialog
        self.id = "redial"

        # store paths to images and labels
        opt["datafile"], movies_with_mentions_path = _path(opt)

        with open(movies_with_mentions_path, "r") as f:
            reader = csv.reader(f)
            self.id2movie = {
                int(row[0]): row[1:] for row in reader if row[0] != "movieId"
            }

        super().__init__(opt, shared)

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
                    yield (source_text, [target_text], None, None, None), new_episode
                    new_episode = False


class DefaultTeacher(RedialTeacher):
    pass
