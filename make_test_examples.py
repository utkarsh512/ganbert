# Author: Utkarsh Patel
#
# preparing test.tsv

from Preprocess import Preprocess
import pandas as pd
import numpy as np 

import argparse
import os
from tqdm import tqdm

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--indir", default=None, type=str, required=True, help="directory of json files containing comments")
  parser.add_argument("--outdir", default=None, type=str, required=True, help="directory to store test.tsv file")
  args = parser.parse_args()
  pr = Preprocess()

  reader = pd.read_json(args.indir, lines=True, compression=None)
  comments = list(reader['body'])
  violated_rule = list(reader['violated_rule'])

  writer_addr = os.path.join(args.outdir, 'test.tsv')
  writer = open(writer_addr, 'w')
  writer.write('label comments\n')

  test_ah = 0
  test_none = 0

  for i in tqdm(range(len(comments)), unit=" comments", desc="comments processed"):
    label = 'NONE'
    test_none += 1
    if violated_rule[i] == 2:
      label = 'AH'
      test_ah += 1
      test_none -= 1
    cur = pr.preprocess(comments[i])
    cur = ' '.join(cur)
    cur = label + ' ' + cur + '\n'
    writer.write(cur)
  
  writer.close()

  print(f'Test set: AH: {test_ah} - NONE: {test_none}')
  
if __name__ == '__main__':
  main()
