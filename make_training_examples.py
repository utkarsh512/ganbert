# Author: Utkarsh Patel
#
# preparing labeled.tsv and unlabeled.tsv

from Preprocess import Preprocess
import pandas as pd
import numpy as np 

import argparse
import os
from tqdm import tqdm

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ratio", default=None, type=float, required=True, help="ratio of training set to be used as labeled examples")
  parser.add_argument("--indir", default=None, type=str, required=True, help="directory of json files containing comments")
  parser.add_argument("--outdir", default=None, type=str, required=True, help="directory to store labeled.tsv and unlabeled.tsv files")
  args = parser.parse_args()
  pr = Preprocess()

  reader = pd.read_json(args.indir, lines=True, compression=None)
  comments = list(reader['body'])
  violated_rule = list(reader['violated_rule'])

  labeled_comments = comments[: int(args.ratio * len(comments))]
  unlabeled_comments = comments[int(args.ratio * len(comments)):]

  labeled_rule = violated_rule[: int(args.ratio * len(violated_rule))]
  unlabeled_rule = violated_rule[int(args.ratio * len(violated_rule)):]

  writer_addr = os.path.join(args.outdir, 'labeled.tsv')
  writer = open(writer_addr, 'w')
  writer.write('label comments\n')

  labeled_ah = 0
  labeled_none = 0

  print('Preparing labeled.tsv...')

  for i in tqdm(range(labeled_comments), unit=" comments", desc="comments processed"):
    label = 'NONE'
    labeled_none += 1
    if labeled_rule[i] == 2:
      label = 'AH'
      labeled_ah += 1
      labeled_none -= 1
    cur = pr.preprocess(labeled_comments[i])
    cur = ' '.join(cur)
    cur = label + ' ' + cur + '\n'
    writer.write(cur)
  
  writer.close()

  writer_addr = os.path.join(args.outdir, 'unlabeled.tsv')
  writer = open(writer_addr, 'w')
  writer.write('label comments\n')

  print('Preparing unlabeled.tsv...')

  for i in tqdm(range(unlabeled_comments), unit=" comments", desc="comments processed"):
    label = 'UNK'
    cur = pr.preprocess(unlabeled_comments[i])
    cur = ' '.join(cur)
    cur = label + ' ' + cur + '\n'
    writer.write(cur)
  
  writer.close()

  print(f'Training set: AH: {labeled_ah} - NONE: {labeled_none}')
  
if __name__ == '__main__':
  main()
