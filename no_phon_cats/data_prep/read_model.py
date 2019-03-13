# -*- coding: utf-8 -*-
"""
Created on Thu Mar 7 13:15:00 2019
@author: Thomas Schatz

"""

import numpy as np
import pandas as pd
import io
import re
import codecs


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


def read_kaldi_phonemap(phones_file, word_position_dependent=True):
    phonemap = dict()
    for line in open(phones_file, 'r', encoding='UTF-8'):
        phone, code = line.strip().split(' ')
        # remove word position markers
        if word_position_dependent and phone[-2:] in ['_I', '_B', '_E', '_S']:
            phone = phone[:-2]
        phonemap[code] = phone
    return phonemap


def get_phone_order(phonemap):
    """
    Output an easily reproducible phone order from a phonemap
    obtained by reading a phones.txt file with k2a.read_kaldi_phonemap
    """
    # remove kaldi disambiguation symbols and <eps> from the phonemap,
    # as those shouldn't be in the phone_order
    codes = list(phonemap.keys())
    for code in codes:
        if re.match(u'#[0-9]+$|<eps>$', phonemap[code]):
            del phonemap[code]

    # order the phones in an easily reproducible way unique is needed
    # since there can be several variants of each phone in the map
    phone_order = list(np.unique(list(phonemap.values())))
    phone_order.sort()  # to guarantee reproducible ordering
    return phone_order


def get_hmm_phone_info(phones_file, word_position_dependent=True):
  return get_phone_order(read_kaldi_phonemap(phones_file, word_position_dependent))


def augment_hmm_state(state, trans_info):
    return (state, *trans_info[state])


def get_folding_plan(hmm_states, reduced_states, reducer):
    plan = {}
    for state_hash in hmm_states:
        rstate = reducer(hmm_states[state_hash])
        rstate_hash = reduced_states.index(rstate)
        if rstate_hash in plan:
            plan[rstate_hash].append(state_hash)
        else:
            plan[rstate_hash] = [state_hash]
    assert set(plan) == set(range(len(reduced_states)))
    return plan


def fold_state(feat_matrix, folding_plan):
    reduced_feats = []
    for rdim in range(len(folding_plan)):
        # a quick hack to handle the fact that we do not include dimensions at the end
        # of the posterior if they are never used (see observed_dimension thing above)
        # this is not ideal (it's not very efficient and could mask some actual errors...)
        dims = folding_plan[rdim]
        dims.sort()
        dims = np.array(dims)
        dims = dims[dims < feat_matrix.shape[1]]
        if len(dims) == 0:
            # not sure if this is necessary, maybe numpy can handle this smartly?
            reduced_feats.append(np.zeros(shape=(feat_matrix.shape[0],)))
        else:
            reduced_feats.append(feat_matrix[:, dims].sum(axis=1))
    reduced_feats = np.column_stack(reduced_feats)
    return reduced_feats


def get_hmm_state_folder(transitions_file):
  # returns a folding function (not a directory)
  # Get the reduced HMM-state that we are interested (replacing both HMM-tied-state and HMM-state).
  # order: phone, word-position, hmm-state, transition-index, state-pdf
  hmm_state_info = get_hmm_state_info(transitions_file)
  reducer = lambda state: (state[0], state[2], state[4])
  reduced_states = list(set([reducer(state) for state in hmm_state_info.values()]))
  reduced_states.sort()  # for reproducibility of ordering
  # to get info about hash for reduced_state: reduced_states.index(hash) -> returns phone, hmm-state-id, pdf-id
  hmm_state_folder = lambda times, feats: (times, fold_state(feats, get_folding_plan(hmm_state_info, reduced_states, reducer)))
  return hmm_state_folder, reduced_states

  