# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:13:48 2018

@author: Thomas Schatz

Collect modelreps for each phone with its context and save the whole thing to disk.
Callable from command-line.


Getting model representations of interest phone by phone with associated context and storing results to disk.
Both the extracted representation and the context information are customizable (cf. modelrep_f and context_f).

Given a corpus (cf. read_corpus.py), raw model representations for the utterances
in that corpus (cf. read_feats.py), a function to extract desired model representations
for each phone (cf. model_rep.py) and a function to extract desired context representations
for each phone (cf. context.py), assemble in a dataframe contexts, phones and model representations
for each phone in the corpus and optionally save it to disk
"""


import pandas as pd
import no_phon_cats.data_prep.read_corpus as corpus_reader
import no_phon_cats.data_prep.read_feats as feats_reader
import no_phon_cats.phone_rep.context as contexts
import no_phon_cats.phone_rep.model_rep as modelreps
import argparse
import io
import yaml


def get_phone_reps(corpus, get_utt_features, modelrep_f, context_f, verbose=0):
    """
    Returns a dataframe containg context, phone and model representation for each phone
    in the corpus. Contexts are defined from corpus data associated to each phone using context_f.
    Model representations are defined from corpus data associated to each phone and raw model
    representations for the utterance using modelrep_f. Raw model representations for any given
    utterance are returned by calling get_utt_features with the utterance name.
    
    Params:
        corpus: pandas.DataFrame. Containing info for each phone in the corpus
        get_utt_features: utt_id:string -> 
                  (utt_times:nb_frames array,
                   utt_feats:(nb_frames x rep_dim) array). 
        context_f: phone_info:pandas.Series -> context:hashable object).
        modelrep_f: (phone_info:pandas.Series,
                     utt_times:nb_frames array,
                     utt_feats:(nb_frames x rep_dim) array)
                    -> modelrep:hashable object.     
    """
    contexts, phones, modelreps = [], [], []
    if verbose:
        print("Processing {} utterances".format(len(corpus.groupby('utt')))) 
    for i, (utt_id, utt_info) in enumerate(corpus.groupby('utt')):
        if verbose and i % 1000 == 0:
            print("Done {} utterances".format(i))
        utt_times, utt_feats = get_utt_features(utt_id)
        contexts.append(utt_info.apply(context_f, axis=1))
        phones.append(utt_info['phone'])
        modelrep_f_utt = lambda phone_info: modelrep_f(phone_info, utt_times, utt_feats)
        modelreps.append(utt_info.apply(modelrep_f_utt, axis=1))
    data = {'context': pd.concat(contexts), 'phone': pd.concat(phones), 'modelrep': pd.concat(modelreps)}
    return pd.DataFrame(data)


def collect_phone_reps(corpus, models, get_utt_features, modelrep_f, context_f, verbose=0):
    # Get phone reps for several different models and concatenate results in a DataFrame
    data = []
    for model in models:
        print(model)
        model_data = get_phone_reps(corpus, get_utt_features[model], modelrep_f[model], context_f[model], verbose=1)
        model_data['model'] = [model] * len(model_data)
        data.append(model_data)
    data = pd.concat(data)
    return data


def read_conf(conf_file):
    # Get paths to all relevant files
    with io.open(conf_file, 'r') as fh:
        files = yaml.load(fh)
    return files


def collect(corpus_name, corpus_conf, feats_conf, out=None,
            rep_type='dominant_unit_around_center',
            context_type='word_phone_spk_context',
            dur=50, verbose=False):
  """
  Get phones and associated model representations for each context and save to disk
  
  Input:
    dur: int (Dominant units around frame center not fused, +/- dur ms from frame center)
  """
  corpus_files, corpus = corpus_reader.read(corpus_name, corpus_conf, verbose=verbose)

  feat_files = read_conf(feats_conf)  
  # Prepare features access
  # This gives us functions for each of GMM, HMM,
  # HMM-tied-state and HMM-state that take as input a utt-id and return the utt features in dense
  # nb_frames x observed_feat_dim matrix format along with timestamps associated with each frame.
  # Be careful about available memory, as this loads sparse kaldi posterior text files in RAM
  get_utt_features = feats_reader.get_features_getter(feat_files)

  models = ['GMM', 'HMM-phone', 'HMM-state', 'HMM-tied-state']

  # model rep. functions
  if rep_type == 'dominant_unit_around_center':
    rep_f = lambda *args: modelreps.dominant_unit_around_center(dur, *args)
  else:
    raise ValueError('Unknown model representation type {}'.format(rep_type))
  modelrep_f = {model: rep_f for model in models}
  # context functions
  if context_type == 'word_phone_spk_context':
    context_f = {model: contexts.word_phone_spk_context for model in models} # phone_context
  else:
    raise ValueError('Unknown context type {}'.format(context_type))
  
  data = collect_phone_reps(corpus, models, get_utt_features, modelrep_f, context_f, verbose=verbose)

  # Do some post processing:
  # getting context into named column for efficient CSV parsing
  context_cols = contexts.word_phone_spk_context_cols()
  for col_name, col_data in zip(context_cols, zip(*data['context'])):
      data[col_name] = col_data
  del data['context']
  if not(out is None):
    # save to csv
    data.to_csv(out)
  else:
    return data
  # We do **not** get complementary info about the HMM states in the representation at this stage
  #  hmm_state_info = get_hmm_state_info(model_files['HMM-transitions'])
  #  modelrep_f['HMM-state'] = lambda *args: augment_hmm_state(rep_f(*args), hmm_state_info)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_name')
    parser.add_argument('corpus_conf')
    parser.add_argument('feats_conf')
    parser.add_argument('out')
    parser.add_argument('--dur', type=int, default=50)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    collect(args.corpus_name, args.corpus_conf, args.feats_conf,
            out=args.out, dur=args.dur, verbose=args.verbose)
