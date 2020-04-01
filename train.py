'''
 * User: Hojun Lim
 * Date: 2020-04-01
'''

import yaml

with open('config.yml') as f:

    docs = yaml.load_all(f, Loader=yaml.FullLoader)

    for doc in docs:
        for k, v in doc.items():
            print(k,v)
            for v_s in v:
                print(v_s.get('123d'))
