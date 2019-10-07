import random
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Evaluate a model')
    parser.add_pytorch_datateacher_args()
    # Get command line arguments
    parser.set_defaults(datatype='valid')
    return parser





if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args(print_args=False)

    # movie_entities = ['<http://dbpedia.org/resource/Star_Wars_(film)>']
    # movie_entities = ['<http://dbpedia.org/resource/Beauty_and_the_Beast_(2017_film)>']
    # movie_entities = ['<http://dbpedia.org/resource/Forrest_Gump>']
    # movie_entities = ['<http://dbpedia.org/resource/The_Shining_(film)>']
    movie_entities = ['<http://dbpedia.org/resource/The_Avengers_(2012_film)>']

    # movie_entities = ['<http://dbpedia.org/resource/Iron_Man_(2008_film)>']
    # movie_entities = ['<http://dbpedia.org/resource/The_Godfather_(film_series)>']
    # movie_entities = ['<http://dbpedia.org/resource/Harry_Potter_and_the_Goblet_of_Fire_(film)>', '<http://dbpedia.org/resource/Harry_Potter_(character)>']
    # movie_entities = ['<http://dbpedia.org/resource/The_Lord_of_the_Rings:_The_Fellowship_of_the_Ring>']
    # movie_entities = ['<http://dbpedia.org/resource/Back_to_the_Future>']


    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    entity2entityId = pkl.load(open('data/redial/entity2entityId.pkl', 'rb'))

    up, _ = agent.model.kbrd.user_representation([
        list(map(lambda x: entity2entityId[x], movie_entities))
    ])

    up_bias = agent.model.user_representation_to_bias_2(
        F.relu(agent.model.user_representation_to_bias_1(up))
    ).squeeze()
    _, idxs = torch.topk(up_bias, 100)
    idxs = idxs[idxs > 100]

    top_words = agent.dict.vec2txt(idxs).split()
    top_words = list(filter(lambda x: not x.startswith('@'), top_words))
    print(top_words)

