import numpy as np
import pickle as pkl
from collections import defaultdict


def get_ripple_set(kg, original_set, n_hop, n_memory):
    """Return: [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]"""

    ripple_set = []

    for h in range(n_hop):
        memories_h = []
        memories_r = []
        memories_t = []

        if h == 0:
            tails_of_last_hop = original_set
        else:
            tails_of_last_hop = ripple_set[-1][2]

        for entity in tails_of_last_hop:
            for tail_and_relation in kg[entity]:
                memories_h.append(entity)
                memories_r.append(tail_and_relation[0])
                memories_t.append(tail_and_relation[1])

        # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
        # this won't happen for h = 0, because only the items that appear in the KG have been selected
        # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
        if len(memories_h) == 0:
            ripple_set.append(ripple_set[-1])
        else:
            # sample a fixed-size 1-hop memory for each user
            replace = len(memories_h) < n_memory
            indices = np.random.choice(len(memories_h), size=n_memory, replace=replace)
            memories_h = [memories_h[i] for i in indices]
            memories_r = [memories_r[i] for i in indices]
            memories_t = [memories_t[i] for i in indices]
            ripple_set.append((memories_h, memories_r, memories_t))

    return ripple_set


DBPEDIA_PATH = "dbpedia/mappingbased_objects_en.ttl"
ID2ENTITY_PATH = "data/redial/id2entity.pkl"


def load_kg(path):
    kg = defaultdict(list)
    with open(path) as f:
        for line in f.readlines():
            tuples = line.split()
            if tuples and len(tuples) == 4 and tuples[-1] == ".":
                h, r, t = tuples[:3]
                # TODO: include property/publisher and subject/year, etc
                if "ontology" in r:
                    kg[h].append((r, t))
    return kg


if __name__ == "__main__":
    kg = load_kg(DBPEDIA_PATH)
    id2entities = pkl.load(open(ID2ENTITY_PATH, "rb"))

    entity_set = defaultdict(int)
    relation_set = defaultdict(int)
    for movieId in id2entities:
        entity = id2entities[movieId]
        if entity is None or kg[entity] == []:
            continue

        ripple = get_ripple_set(kg, [entity], 2, 32)

        for ripple_hop in ripple:
            memories_h, memories_r, memories_t = ripple_hop
            for h in memories_h:
                entity_set[h] += 1
            for r in memories_r:
                relation_set[r] += 1
            for t in memories_t:
                entity_set[t] += 1

    # print number of entities and relations and their average frequency
    print(len(entity_set), np.mean([entity_set[k] for k in entity_set]))
    print(len(relation_set), np.mean([relation_set[k] for k in relation_set]))

