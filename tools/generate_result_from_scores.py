import argparse

from collections import defaultdict
from statistics import mean
from sys import meta_path

import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument('scores')
parser.add_argument('out')
args = parser.parse_args()

scores = pd.read_csv(args.scores, header=None)

obs_ids = scores.iloc[:, 0]

dropped = 0
total = 0
with open(args.out, 'w') as f:
    f.write('observation_id,class_id\n')
    for id in obs_ids:
        scrs = np.array(scores[scores.iloc[:, 0] == id].iloc[0, 1:])
        scrs = torch.softmax(torch.from_numpy(scrs), dim=0).numpy()
        max_score = np.max(scrs)
        cls_id = np.argmax(scrs)
        if max_score < 0.09:
            cls_id = -1
            dropped += 1
        total += 1
        f.write(f'{id},{cls_id}\n')
print(f'dropped {dropped} out of {total}')


