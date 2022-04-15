

import json

path = '/home/dylan/placeschallenge/instancesegmentation/imgCatIds.json'

with open(path) as f:
	dict = json.load(f)

outarr = ['background'] * 101

for category in dict['categories']:
	outarr[category['id']] = category['name']

breakpoint()
