import os
import sys
import json
import numpy as np
from progressbar import ProgressBar

cat = sys.argv[1]
split = sys.argv[2]

with open('%s.%s.json' % (cat, split), 'r') as fin:
    ids = [d['anno_id'] for d in json.load(fin)]

with open('%s-%s-psg.txt' % (cat, split), 'w') as fout:
    for i in range(len(ids)):
        fout.write('%s\n' % ids[i])

