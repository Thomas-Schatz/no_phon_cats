# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:13:48 2018

@author: Thomas Schatz

Function to single out certain phones+contexts for analysis based on various criteria.
"""


import numpy as np


# select some nice context+phones
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
    gpby_cols = ['phone', 'model', 'spk', 'word', 'word_trans', 'phone_pos', 'prev_phone', 'next_phone']
    groups = data.groupby(gpby_cols, as_index=False).size().to_frame('size').reset_index()
    del groups['model']
    l = len(groups)
    groups = groups.drop_duplicates()  # duplicated for each model
    assert len(groups) == l//4  # ad hoc
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