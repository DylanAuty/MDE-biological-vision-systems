# npy_convert_to_int32.py
# Converts every .npy file in the target directory to int32.

import os
import glob
import argparse
import numpy as np
from tqdm import tqdm


def main(path):
	print(path)
	i = 0
	for root, dirs, files in os.walk(path):
		for file in files:
			if file.endswith(".npy"):
				print(i)
				i += 1
				full_path = os.path.join(root, file)
				tmp = np.load(full_path)
				np.save(full_path, tmp.astype(np.int32))

	return

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, help='Path to recursively convert all npy files in to int32')

	args = parser.parse_args()

	main(args.path)

