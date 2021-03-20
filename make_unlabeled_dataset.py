# Author: Utkarsh Patel
#
# preparing unlabeled.tsv

from Preprocess import Preprocess
import pandas as pd
import numpy as np 

import argparse
import os
from tqdm import tqdm

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--indir", default=None, type=str, required=True, help="directory of json files containing comments")
  parser.add_argument("--outdir", default=None, type=str, required=True, help="directory to store unlabeled.tsv files")
  args = parser.parse_args()
  pr = Preprocess()

  reader = pd.read_json(args.indir, lines=True, compression=None)
  comments = list(reader['body'])[:40000]

  writer_addr = os.path.join(args.outdir, 'unlabeled.tsv')
  writer = open(writer_addr, 'w')
  writer.write('label comments\n')

  for i in tqdm(range(len(comments)), unit=" comments", desc="comments processed"):
    cur = pr.preprocess(comments[i])
    cur = ' '.join(cur)
    cur = 'UNK ' + cur + '\n'
    writer.write(cur)
  
  writer.close())
  
if __name__ == '__main__':
  main()
