#!/usr/bin/env python3

# This program receives 2 text files as input and calculates the accuracy of the
# predictions (argument 1) with respect to the reference/gold (argument 2). The 
# resulting score ranges from 0% (bad) to 100% (perfect).

import sys
import random
import argparse

# Treat list of arguments in command line: must be valid filenames
parser = argparse.ArgumentParser(description='Guess the polaity of a text. \
Inputs are lists of sentences, one per line. Everything after tabulation (TAB) \
is considered as a category label.')
parser.add_argument('predfile', type=argparse.FileType('r', encoding='UTF-8'),
                   help='Prediction text file, with one sentence per line, UTF-8')
parser.add_argument('goldfile', type=argparse.FileType('r', encoding='UTF-8'),
                   help='Gold text file, with one sentence per line, UTF-8')                   
args = parser.parse_args()

total = tp = 0
for (predline,goldline) in zip(args.predfile, args.goldfile):
  try :
    predtext, predcat = predline.strip().split("\t", 1)
    goldtext, goldcat = goldline.strip().split("\t", 1)
  except ValueError :
    print(f"Error line {total+1}: file not well formatted", file=sys.stderr)    
    sys.exit(-1)
  #import pdb
  #pdb.set_trace()
  if predtext != goldtext :
    print(f"Error line {total+1}: pred and gold files not aligned!", file=sys.stderr)    
    sys.exit(-1)
  if goldcat == predcat :
    tp += 1
  total += 1
acc = 100.0 * (tp / total)
print("Predictions file: {}".format(args.predfile.name))
print("Gold/reference file: {}".format(args.goldfile.name))
print(f"Accuracy on all words: {acc:0.2f}% ({tp}/{total})")
