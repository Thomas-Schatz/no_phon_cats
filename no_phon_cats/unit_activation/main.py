# -*- coding: utf-8 -*-
"""
Created on Thu Mar 7 13:15:00 2019
@author: Thomas Schatz

"""


import no_phon_cats.data_prep.read_corpus as corpus_reader
import no_phon_cats.data_prep.read_feats as feats_reader
import no_phon_cats.data_prep.read_model as model_reader
import no_phon_cats.unit_activation.duration as duration
import no_phon_cats.unit_activation.sharpness as sharpness
import argparse
import io
import yaml


def read_conf(conf_file):
    # Get paths to all relevant files
    with io.open(conf_file, 'r') as fh:
        files = yaml.load(fh)
    return files


def control_utt_sils(corpus):
  # Getting beginning and end of actual speech (as opposed to silence/noise) for each utterance
  # As a control, we also keep track of the subset of all utterances where there is no within-utterance silences at all
  all_segments, segments_nosil = [], []
  for utt, utt_df in corpus.groupby('utt'):
      start, stop = list(utt_df['start']), list(utt_df['stop'])
      all_segments.append((utt, start[0], stop[-1]))
      if start[1:] == stop[:-1]:
          segments_nosil.append((utt, start[0], stop[-1]))
  dur = []
  for utt, start, stop in all_segments:
      dur.append(stop-start)                               
  print("Nb utts: {}".format(len(all_segments)))
  print("Nb utts without inner silence/noise: {}".format(len(segments_nosil)))
  print("Shortest utt: {} sec.".format(min(dur)))
  print("Longest utt: {} sec.".format(max(dur)))
  return all_segments, segments_nosil


def collect(corpus_name, corpus_conf, feats_conf, model_conf, out=None,
            max_nb_frames=3, frame_dur=.01, min_thr=.05, duration_test_type='basic',
            verbose=False):
  # With max_nb_frames=3 True False False False True would work
  ###
  # Load conf
  ###
  corpus_files, corpus = corpus_reader.read(corpus_name, corpus_conf, verbose=verbose)
  feat_files = read_conf(feats_conf)  
  model_files = read_conf(model_conf) 
  # Control for utterance onset/offset silence and utterance-medial silence
  # e.g. for WSJ: 
  #  1178/4855 total utterances have no within-utterance silence
  #  (effective) utterance duration range is from .47 to 15.91 seconds
  #  (so no issue with short sentences there)
  all_segments, segments_nosil = control_utt_sils(corpus)
  ###
  # Prepare features access
  ###
  # folding function for HMM state (not a string)
  hmm_state_folder, hmm_reduced_state_info = model_reader.get_hmm_state_folder(model_files['HMM-transitions'])
  # This gives us functions for each of GMM, HMM,
  # HMM-tied-state and HMM-state that take as input a utt-id and return the utt features in dense
  # nb_frames x observed_feat_dim matrix format along with timestamps associated with each frame.
  # Be careful about available memory, as this loads sparse kaldi posterior text files in RAM
  get_utt_features = feats_reader.get_features_getter(feat_files,
                                                      include_tied_state_HMM=False,
                                                      HMM_states_folder=hmm_state_folder)
  ###
  # Duration
  ###
  # For selected segments (utterances), for each model and each feature dimension for that model,
  # get average duration with which that feature gets activated (this is different from average duration with
  # which a unit remains dominant).
  # We could average first on individual sentences and then over all sentences.
  # Emmanuel tried it without seeing anything different, so here we just average over everything ignoring the grouping
  # into sentences.
  models = ['GMM', 'HMM-phone', 'HMM-state']  #'HMM-tied-state']
  if duration_test_type == 'basic':
    get_dur = lambda feats: duration.get_duration_basic(feats, frame_dur)
  elif duration_test_type == 'conservative':
    get_dur = lambda feats: duration.get_duration_conservative(feats, max_nb_frames, frame_dur, min_thr)
  else:
    raise ValueError('Unsupported type of duration test: {}'.format(duration_test_type))
  durs = {}  # duration and number of dominant episodes for each feature dimension
  for model in models:
      print(model)
      durs[model, 'all utts'] = duration.activation_duration(model, all_segments,
                                                    get_utt_features,
                                                    get_duration=get_dur)
      durs[model, 'no-sil utts'] = duration.activation_duration(model, segments_nosil,
                                                       get_utt_features,
                                                       get_duration=get_dur)
  
  ###
  # Sharpness/activation level
  ###
  # Measure activation sharpness feature by feature (how activated does that feature get when it's dominant).
  # Save individual values.
  models = ['HMM-phone', 'GMM', 'HMM-state'] #, 'HMM-tied-state']
  activation_levels = {}  # duration and number of dominant episodes for each feature dimension
  for model in models:
      print(model)
      activation_levels[model, 'all utts'] = sharpness.activation_sharpness(model, all_segments,
                                                                            get_utt_features)
      activation_levels[model, 'no-sil utts'] = sharpness.activation_sharpness(model, segments_nosil,
                                                                               get_utt_features)
      
  ###
  # Save results
  ###   
  # A text file with a line by feature and a list of a numbers on each line (space-separated)
  # Interpretation of the features dimensions when available: reduced phone states + phones                                                              
  hmm_phone_info = get_hmm_phone_info(model_files['HMM-phones'])

  if not(out is None):
    with open(out + '_hmm_phone_info.txt') as fh:
      fh.print(" ".join(hmm_phone_info) + '\n')
    with open(out + '_hmm_state_info.txt') as fh:
      for s1, s2, s3 in hmm_reduced_state_info:
        fh.print('{} {} {}\n'.format(s1, s2, s3))
    for model, condition in durs:
      with open(out + 'duration_{}_{}.txt'.format(model, condition)) as fh:
        for feat_durs in durs[model, condition]:
          fh.print(" ".join(map(str, feat_durs)) + '\n')
    for model, condition in activation_levels:
      with open(out + 'sharpness_{}_{}.txt'.format(model, condition)) as fh:
        for feat_sharps in activation_levels[model, condition]:
          fh.print(" ".join(map(str, feat_sharps)) + '\n')
  else:
    return durs, activation_levels, hmm_phone_info, hmm_reduced_state_info


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_name')
    parser.add_argument('corpus_conf')
    parser.add_argument('feats_conf')
    parser.add_argument('model_conf')
    parser.add_argument('out')
    parser.add_argument('--max_nb_frame', type=int, default=3)
    parser.add_argument('--frame_dur', type=float, default=.01)
    parser.add_argument('--min_thr', type=float, default=.05)
    parser.add_argument('--duration_test_type', default='basic')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    collect(args.corpus_name, args.corpus_conf, args.feats_conf, args.model_conf,
            out=args.out, max_nb_frames=args.max_nb_frame, frame_dur=args.frame_dur,
            min_thr=args.min_thr, duration_test_type=args.duration_test_type,
            verbose=args.verbose)
