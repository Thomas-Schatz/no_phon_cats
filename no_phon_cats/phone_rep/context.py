# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:13:48 2018

@author: Thomas Schatz

Functions to extract phone contexts:
- Input is row of corpus DataFrame for target phone
- Output needs to be hashable

As long as memory is not an issue it makes sense to go for the most detailed context representations,
as it is always possible to derived less detail representations from them later on.
"""

def word_phone_spk_context_cols():
    return ('word', 'word_trans', 'phone_pos', 'prev_phone', 'next_phone', 'spk',
            'utt', 'word_start', 'word_stop', 'start', 'stop')


def word_phone_spk_context(phone_info):
    context = tuple([phone_info[col] for col in word_phone_spk_context_cols()])
    return context


def word_context(phone_info):
    return phone_info['word'], phone_info['word_trans'], phone_info['phone_pos']


def word_context_cols():
    return ('word', 'word_trans', 'phone_pos')


def word_spk_context(phone_info):
    return phone_info['word'], phone_info['word_trans'], phone_info['phone_pos'], phone_info['spk']


def word_spk_context_cols():
    return ('word', 'word_trans', 'phone_pos', 'spk')


def phone_context(phone_info):
    return phone_info['prev_phone'], phone_info['next_phone']


def phone_context_cols():
    return ('prev_phone', 'next_phone')


def phone_spk_context(phone_info):
    return phone_info['prev_phone'], phone_info['next_phone'], phone_info['spk']


def phone_spk_context_cols():
    return ('prev_phone', 'next_phone', 'spk')

# what about word_pos for utterance final, initial etc.? Or we could add syllable position etc.