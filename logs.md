---
title: "EMNLP'19"
author: Qibin Chen
geometry: "left=1in,right=1in,top=1in,bottom=1in"
---

# Plans

1. Baseline
    - [x] reproduce baseline final results
    - [ ] better evaluation metrics
        - hits@ for rec
        - bleu for dialog
    - [x] ParlAI
    - ~~gensen to bert~~
#. Entity linking
    - [x] mentions in dialog: DBpedia-spotlight
    - [ ] align redial movies with dbpedia entity
    - [ ] current dbpedia/mappingbased_objects_en.ttl only inclues dbo (ontology); publisher (dbp, property)  and year (dct:subject) information are missing!!
#. ~~Wikipedia retriever~~
    - ~~<https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/README.md>~~
#. ~~Wikipedia for movie entities (rec to text)~~
    - [x] ~~align redial movies with wikimovies~~
    - ~~memory networks / bert / soft template~~
#. Content-based recommendation (text to rec)
    - [ ] GraphSAGE, use ComplEx embedding on dbpedia as initialization
    - ~~Autorec, NCF, CDL, DeppCoNN, CKE~~
    - ~~To combine content-based CF with user-based recommender.~~ (the nips authors tried but failed)

# Logs

## Apr. 4

- data, baseline

## Apr. 5 – 7

- survey papers on conversational recommendation (2002-2016)

## Apr. 8 Running baselines

- changed autorec layers to \[512, 256, 128\], relu, no improvement
- Gensen: Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning <https://arxiv.org/abs/1804.00079>
- HRNN output in sentiment analysis: 512 fixed vector, can be changed to bert; movie occurrence indicator is catted to word embed; sender indicator appended to sen embed
- lacking proper metrics for generated responses
- ml rating (0.5 to 5) is scaled to \[0, 1\]; binarizing to 0 and 1 increases the error

## Apr. 9 Wikipedia

- Movies in REDIAL are linked to Wikimovies. For Redial 3678 matched out of 6924. For MovieLens, 11835 matched out of 59944.
- In autorec validation/test, use others to predict one-left-out in a conversation.
- The baseline didn't distinguish negative and unknown in autorec input.
- Add bert embeddings to autorec; no improvement

## Apr. 10 Knowledge

- wikipedia to kg
- dbpedia spotlight docker
- download dbpedia **mappingbased\_objects\_en** \~18M triples, \~6M entities
- \* Augmented graph <http://arxiv.org/abs/1903.10245>

## Apr. 11 – 12

- Biggraph ComplEx embeddings for DBpedia
    - test pos\_rank: 57.4009 , mrr: 0.586123 , r1: 0.493711 , r10: 0.756012 , r50: 0.858827 , auc: 0.950244 , count: 47070
    - train pos\_rank: 6.03484 , mrr: 0.538354 , r1: 0.407232 , r10: 0.837359 , r50: 0.996619 , auc: 0.994823 , count: 1.86519e+07
    - compared with <https://github.com/dbpedia/embeddings/tree/master/gsoc2017-nausheen>
- Integration into ParlAI
    - transformer baseline:
        - best validation
            - valid:{'exs': 5573, 'accuracy': 0.01435, 'f1': 0.1627, 'bleu': 0.01297, 'lr': 1, 'num_updates': 35890, 'loss': 337.4, 'token_acc': 0.3347, 'nll_loss': 3.644, 'ppl': 38.24}
            - test:{'exs': 7087, 'accuracy': 0.01524, 'f1': 0.1625, 'bleu': 0.01713, 'lr': 1, 'num_updates': 35890, 'loss': 432.9, 'token_acc': 0.3298, 'nll_loss': 3.655, 'ppl': 38.68}
    - seq2seq baseline:
        - best validation
            - valid:{'exs': 5573, 'accuracy': 0.00969, 'f1': 0.155, 'bleu': 0.01154, 'lr': 1, 'num\_updates': 102659, 'loss': 378.4, 'token\_acc': 0.3255, 'nll\_loss': 4.083, 'ppl': 59.35}
            - test:{'exs': 7087, 'accuracy': 0.01115, 'f1': 0.1598, 'bleu': 0.01403, 'lr': 1, 'num\_updates': 102659, 'loss': 476.9, 'token\_acc': 0.3223, 'nll\_loss': 4.025, 'ppl': 56.01}

## Apr. 13 – 14

- Recommendation metric for nips baseline
    - hit@10, ....
    - baseline results with movielens pretraining & use reply @ best valid point: recall@1 = 0.03138, recall@10 = 0.15231, recall@50 = 0.32957
    - baseline results without pretraining & use reply @ best valid point: recall@1 = 0.0149505051535871, recall@10 = 0.10296969078477396, recall@50 = 0.22435962853352384
- Match redial movies to dbpedia
    - 6272 of 6924 movies are matched!!
- [ ] match movieslens to dbpedia
    - we don't consider movielens dataset for now?

## Apr. 15

- baseline in ParlAI

## Apr. 16
