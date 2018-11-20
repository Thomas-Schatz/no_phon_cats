# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:13:48 2018

@author: Thomas Schatz

Augmenting model representations with additional information

(currently only functions to add information about HMM-states for kaldi HMM models)
"""


import pandas as pd
import io


def read_kaldi_transitions(transitions_file):
    """
    This provides information regarding HMM states (phone, hmm-state, hmm-transition-index, pdf).
    There is no information about which contexts are allowed around each state (triphone stuff),
    the decision tree should be consulted for that.
    
    All numeric indices are converted to be 0-based. (transition-id and transition-state were 1-based)
    """
    # I think pdf-count is related to the .occs file passed as argument to show-transitions
    # and is some kind of frequency statistics based on training data.
    cols = ['transition-id', 'pdf-count',
            'transition-state',
            'phone', 'word-position',
            'hmm-state', 'state-pdf', 'transition-index',]
    data = {e: [] for e in cols}
    with io.open(transitions_file, 'r') as fh:
        for line in fh:
            if line[:16] == 'Transition-state':
                t_ix = 0
                t_s, rest = line.strip().split(':')
                t_s = int(t_s[16:])
                tokens = rest.split(" ")
                phone, hmm_s, pdf_s = tokens[3], int(tokens[6]), int(tokens[9])
                if "_" in phone:  # ad hoc
                    phone, wp = phone.split("_")
                else:
                    wp = "None"
                continue
            else:
                assert line[:14] == ' Transition-id', line
                tokens = line.split(" ")
                t_id, pdf_c = int(tokens[3]), int(tokens[11])
                data['transition-id'].append(t_id - 1)  # convert to 0-based indexing
                data['pdf-count'].append(pdf_c)           
                data['transition-state'].append(t_s - 1)  # convert to 0-based indexing
                data['phone'].append(phone)
                data['word-position'].append(wp)
                data['hmm-state'].append(hmm_s)
                data['state-pdf'].append(pdf_s)
                data['transition-index'].append(t_ix)
                t_ix = t_ix+1
    return pd.DataFrame(data)        


def get_hmm_state_info(transitions_file):
    transitions = read_kaldi_transitions(transitions_file)
    trans_info = {row['transition-id']: (row['phone'],
                                         row['word-position'],
                                         row['hmm-state'],
                                         row['transition-index'],
                                         row['state-pdf'],)
                  for _, row in transitions.iterrows()}
    return trans_info


def augment_hmm_state(state, trans_info):
    return (state, *trans_info[state])