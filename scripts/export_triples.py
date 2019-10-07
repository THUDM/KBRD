import pickle as pkl

if __name__ == "__main__":
    subkg = pkl.load(open('data/redial/subkg.pkl', 'rb'))
    entity2entityId = pkl.load(open('data/redial/entity2entityId.pkl', 'rb'))
    entityId2entity = dict([(entity2entityId[k], k) for k in entity2entityId])
    relation2relationId = pkl.load(open('data/redial/relation2relationId.pkl', 'rb'))
    relationId2relation = dict([(relation2relationId[k], k) for k in relation2relationId])

    triples = []
    for headId in subkg:
        for relationId, tailId in subkg[headId]:
            head = entityId2entity[headId]
            relation = relationId2relation[relationId]
            tail = entityId2entity[tailId]
            if relation == 'self_loop':
                continue
            triples.append(f"{head} {relation} {tail} \n")

    print(len(triples))
    with open('triples.txt', 'w') as f:
        f.writelines(triples)

