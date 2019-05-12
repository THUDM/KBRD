import argparse
import json
import numpy as np
from scipy import stats


def read(name, suffix):
    results = []
    for i in range(args.num):
        fname = name + f'_{i}.{suffix}'
        with open(fname, 'r') as f:
            result = f.readlines()[-1]
            result = result[result.index('{'):].replace("'", '"')
            result = json.loads(result)
        results.append(result[args.metric])
    print(results)
    print(np.mean(results))
    print(np.std(results))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name-1', type=str)
    parser.add_argument('--name-2', type=str)
    parser.add_argument('--num', type=int)
    parser.add_argument('--metric', type=str)
    parser.add_argument('--suffix', type=str, default='test')
    args = parser.parse_args()

    results_1 = read(args.name_1, args.suffix)
    results_2 = read(args.name_2, args.suffix)
    print(stats.ttest_ind(results_1, results_2))

