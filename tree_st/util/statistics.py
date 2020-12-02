'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

from collections import Counter

def print_counter_stats(c: Counter, title='', top_n=10, file=None):
    numeric_keys = [k for k in c.keys() if isinstance(k, int)]
    if numeric_keys:
        print(f'Avg. {title}', f'{sum(k * c[k] for k in numeric_keys) / sum(c[k] for k in numeric_keys):3.2f}', sep='\t', file=file)
        print(f'Min {title}', f'{min(numeric_keys):3.0f}', sep='\t', file=file)
        print(f'Max {title}', f'{max(numeric_keys):3.0f}', sep='\t', file=file)
    total = sum(c.values())
    print(f'Total {title} types', f'{len(c):10.0f}', sep='\t', file=file)
    print(f'Total {title} tokens', f'{total:10.0f}', sep='\t', file=file)
    for k, v in c.most_common(top_n):
        print(k, f'{v:10.0f}', f'[{(100*v)/total:6.2f}%]' if total else '[-------]', sep='\t', file=file)
