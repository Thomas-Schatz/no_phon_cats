# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:51:00 2019
@author: Thomas Schatz

Phone duration statistics from forced alignment
"""

import no_phon_cats.data_prep.read_corpus as corpus_reader
import argparse

###

def get_dur(corpus_name, corpus_conf, out=None, verbose=False):
  corpus_files, corpus = corpus_reader.read(corpus_name, corpus_conf, verbose=verbose)
  corpus['phone_dur'] = corpus['stop']-corpus['start']
  durs = corpus.groupby(['phone', 'spk'], as_index=False)['phone_dur'].mean()
  if out is None:
    return durs
  else:
    durs.to_csv(out)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_name')
    parser.add_argument('corpus_conf')
    parser.add_argument('out')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    get_dur(args.corpus_name, args.corpus_conf, out=args.out, verbose=args.verbose)
