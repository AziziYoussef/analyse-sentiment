#!/usr/bin/env python3

# This program receives text files as input and as an output generates 
# a random category for each line. It is a dummy baseline intended as
# a first system to get familiar with the dataset.

import sys
import random
import argparse

# List of possible categories
CATEGORIES = ['+', '-', '=']

# Treat list of arguments in command line: must be valid filenames
parser = argparse.ArgumentParser(description='Guess the polaity of a text. \
Input is a list of sentences, one per line. Everything after tabulation (TAB) \
will be ignored and replaced by the polarity code.')
parser.add_argument('textfile', type=argparse.FileType('r', encoding='UTF-8'),
                   help='Text file, with one sentence per line, UTF-8')
args = parser.parse_args()


# For each text file in argument list
for text in args.textfile :
  # Print the filename and a random language name
  file_line = text.strip().split("\t")[0]
  print("\t".join([file_line,random.choice(CATEGORIES)]))
