# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:13:48 2018

@author: Thomas Schatz

Function to single out certain phones+contexts (i.e. phonetic categories)
for analysis based on various criteria and sample some items from these.
"""


import numpy as np
import pandas as pd


"""
Select some nice context+phones
"""

def is_in_middle(pos, length):
    middle = (length-1)//2
    if length % 2 == 0:
        res = (pos == middle) or (pos == middle+1)
    else:
        res = pos == middle
    return res


def is_within_word(pos, length):
    res = (pos > 0) and (pos < length-1)
    return res


def select_phones_in_contexts(data, by_spk=True, by_word=True, by_phon_context=True,
                              position_in_word='middle', min_wlen=5, min_occ=10, verbose=0):
    # first group with finest granularity and get rid of duplicates
    gpby_cols = ['phone', 'spk', 'word', 'word_trans', 'phone_pos', 'prev_phone', 'next_phone']
    groups = data.groupby(gpby_cols, as_index=False).size().to_frame('size').reset_index()
    # second filter out any group where the word is too short or the phone position in the word is not adequate
    if position_in_word == 'middle':
        wpos_selector = is_in_middle
    elif position_in_word == 'within-word':
        # not first or last phone of word
        wpos_selector = is_within_word
    else:
        assert position_in_word == 'any'
        wpos_selector = lambda *x: True   
    groups = groups[np.array([len(wtrans)>=min_wlen for wtrans in groups['word_trans']]) &
                    np.array([wpos_selector(pos, len(wtrans)) for pos, wtrans in zip(groups['phone_pos'],
                                                                             groups['word_trans'])])]
    # third group only by required characteristics
    gpby_cols = ['phone']
    if by_spk:
        gpby_cols += ['spk']
    if by_word:
        gpby_cols += ['word', 'word_trans', 'phone_pos']  # same word and same position in word
    if by_phon_context:
        gpby_cols += ['prev_phone', 'next_phone']   
    groups = groups.groupby(gpby_cols, as_index=False)['size'].sum()
    # finally only retain large enough groups
    groups = groups[groups['size'] >= min_occ]
    if verbose:
        print("Found {} appropriate context+phone".format(len(groups)))
    return groups


"""
Find items in corpus associated to each context+phone type
"""
def find_indices(data, **kwargs):
    # Isn't there a pandas built-in to do that?
    # usage example: find_indices(data, model='HMM-state', spk='4aw')
    # Warning: this is iloc style
    ix = [True] * len(data)
    for col in kwargs:
        assert col in data, (col, data.columns)
        ix = ix & (data[col] == kwargs[col])
    return np.where(ix)[0]


def get_context_phone_occs(data, context_phones, verbose=False):
    # return subset of data corresponding to specified context_phones
    # with an additional 'context+phone ID' column containing the index
    # to corresponding row of context_phones
    # (not optimized, if optimization needed, could probably do it directly when selecting
    #  the context+phones)
    data_indices, cp_indices = [], []
    cp_cols = [col for col in context_phones.columns if col != 'size']
    if verbose:
        print("Processing {} context+phone".format(len(context_phones)))
    for i, (row_ind, row) in enumerate(context_phones.iterrows()):
        if verbose and (i % 100 == 0):
            print("Processed {}".format(i))
        col_values = {col: row[col] for col in cp_cols}
        indices = find_indices(data, **col_values)
        data_indices.append(indices)
        cp_indices = cp_indices + [row_ind]*len(indices)
    data_indices = np.concatenate(data_indices)
    cp_data = data.iloc[data_indices].copy()
    cp_data["context+phone ID"] = cp_indices
    return cp_data


"""
Item sampling to keep size of NP-hard minimal hitting set problem reasonable
"""

def select_cp_occs(cp_data, selection_f, seed=0, verbose=False):
    # select random subset of min_occ occurrences of each context+phone
    np.random.seed(seed)
    if verbose:
        print("Processing {} context+phones (items not types)".format(len(cp_data)))
    selected_data = cp_data.groupby("context+phone ID", as_index=False).apply(selection_f)
    return selected_data


def random_select(data, n):
    # Select n rows of data at random without replacement.
    # If data has less than n rows, return an empty pandas.DataFrame.
    # data is a pandas.DataFrame
    if len(data) < n:
        res = pd.DataFrame(columns=data.columns)
    else:
        perm = np.random.permutation(data.index)
        res = data.loc[perm[:n]]
    return res


def random_select_across_spk(data, n):
    # Select n rows of data at random without replacement, such that each row has a different
    # level for column 'spk'. Sampling is stratified: first 10 speakers are randomly selected
    # uniformly across available speakers, then an occurrence for each speaker is selected
    # uniformly at random among available occurrences.
    # If this is not satisficible, return an empty list.
    # data is a pandas.Series
    # (not optimized)
    spks = np.unique(data['spk'])
    if len(spks) >= n:
        indices = []
        perm = np.random.permutation(spks)
        spks = perm[:n]
        for spk in spks:          
            data_spk = data[data['spk'] == spk]
            indices.append(data_spk.index[np.random.randint(len(data_spk))])
        res = data.loc[indices]
    else:
        res = pd.DataFrame(columns=data.columns)
    return res