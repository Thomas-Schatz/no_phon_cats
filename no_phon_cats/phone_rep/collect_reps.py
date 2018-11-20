# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:13:48 2018

@author: Thomas Schatz

Data collection functions

Given a corpus (cf. read_corpus.py), raw model representations for the utterances
in that corpus (cf. read_feats.py), a function to extract desired model representations
for each phone (cf. model_rep.py) and a function to extract desired context representations
for each phone (cf. context.py), assemble in a dataframe contexts, phones and model representations
for each phone in the corpus.
"""


import pandas as pd


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