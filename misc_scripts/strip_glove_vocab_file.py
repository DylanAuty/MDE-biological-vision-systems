# strip_glove_vocab_file.py
# Input glove vocab file (format is "token no_occurrences", e.g. "is 21281347")
# Output just tokens (no occurrences), separated by a newline.


import os, sys
import argparse
import numpy as np
from tqdm import tqdm


def main(args):
	# Read line by line, split by whitespace, take first thing in split, write to file.
	with open(args.vocab_file, 'r') as vocab_file, open(args.output_file, 'w') as output_file:
		for line in vocab_file:
			token = line.split(" ")[0]
			output_file.write(token)
			output_file.write("\n")


def preprocess_validate_args(args):
	# Make everything into an absolute path
	args.vocab_file = os.path.abspath(args.vocab_file)
	args.output_file = os.path.abspath(args.output_file)

	assert os.path.isfile(args.vocab_file)		# Check vocab file exists
	assert not os.path.isdir(args.output_file)	# Check that output file isn't a dir
	
	if(os.path.isfile(args.output_file)):
		choice = None
		print(f"Error: Output file exists ({args.output_file}). Overwrite? [y/n]:", end=' ')
		choice = input().lower()
		while choice not in ["y", "n"]:
			print(f"Please answer y or n [Y/n]", end=' ')
			choice = input().lower()
		if choice == "y":
			print("Output file overwritten.")
			os.remove(args.output_file)
		else:
			sys.exit("File not deleted. Exiting.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="""
		Loads a GloVe-formatted vocab.txt (token, then integer number of occurrences in the corpus), 
		returns just tokens separated by newlines (i.e. strips out occurrence integer).
		""", fromfile_prefix_chars='@', conflict_handler='resolve')

	parser.add_argument("--vocab_file", required=True, type=str, help="Path to vocab.txt file. See help (-h) for expected format.")
	parser.add_argument("--output_file", required=True, type=str, help="Path to output file. Will write a .txt file.")

	args = parser.parse_args()

	preprocess_validate_args(args)	# No news is good news.

	main(args)
