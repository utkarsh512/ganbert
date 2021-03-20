# Author: Utkarsh Patel
#
# preparing labeled.tsv and test.tsv

from Preprocess import Preprocess
import pandas as pd
import numpy as np 

import argparse
import os
from tqdm import tqdm

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--indir", default=None, type=str, required=True, help="directory of json files containing comments")
  parser.add_argument("--outdir", default=None, type=str, required=True, help="directory to store labeled.tsv and test.tsv files")
  args = parser.parse_args()
  pr = Preprocess()

  reader = pd.read_json(args.indir, lines=True, compression=None)
  comments = list(reader['body'])
  violated_rule = list(reader['violated_rule'])

  writer_addr = os.path.join(args.outdir, 'labeled.tsv')
  writer = open(writer_addr, 'w')
  writer.write('label comments\n')

  for i in tqdm(range(4000), unit=" comments", desc="comments processed"):
    label = 'NONE'
    if violated_rule[i] == 2:
      label = 'AH'
    cur = pr.preprocess(comments[i])
    cur = ' '.join(cur)
    cur = label + ' ' + cur + '\n'
    writer.write(cur)
  
  writer.close()

  writer_addr = os.path.join(args.outdir, 'test.tsv')
  writer = open(writer_addr, 'w')
  writer.write('label comments\n')

  for i in tqdm(range(4000), unit=" comments", desc="comments processed"):
    label = 'NONE'
    if violated_rule[i] == 2:
      label = 'AH'
    cur = pr.preprocess(comments[i])
    cur = ' '.join(cur)
    cur = label + ' ' + cur + '\n'
    writer.write(cur)
  
  writer.close()
  
if __name__ == '__main__':
  main()
