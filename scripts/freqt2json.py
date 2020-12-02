'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys
import json

freqt_file = sys.argv[1]

with open('atomic.json') as f:
    atomic = json.load(f)

d = dict(atomic)

with open(freqt_file) as f:
    for i, line in enumerate(f, start=len(d)):
        line = line.strip()
        d[line] = i

print(json.dumps(d, indent=2))
