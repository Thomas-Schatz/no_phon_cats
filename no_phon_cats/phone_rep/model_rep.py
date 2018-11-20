# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:13:48 2018

@author: Thomas Schatz

Functions to extract various model representations of phones:
  - Input is row of corpus DataFrame for target phone and
    model representations with associated timestamps for the utterance
  - Output needs to be hashable
"""


import numpy as np


def raw_representation(on, off, utt_times, utt_feats, return_times=False):
    # if return_times is true, return phone times relative to phone start
    t = (utt_times >= on) & (utt_times <= off)
    feats = utt_feats[t, :]
    if feats.shape[0] == 0:
        print('No features found!')
    if return_times:
        times = utt_times[t] - on
        return times, feats
    else:
        return feats


def fuse(seq):
    # inspired from Emmanuel's get_uniq
    # replace contiguous duplicates with a single instance, e.g. 1 1 1 1 2 0 3 3 3 1 3 -> 1 2 0 3 1 3
    shifted_seq = np.concatenate([[np.nan], seq[:-1]])
    onsets = seq != shifted_seq
    return seq[onsets]


def dominant_unit_seq(phone_info, utt_times, utt_feats, fuse=True):
    # sequence of most probable units
    # if fuse=True: if the same unit is dominant at several consecutive frames, it will only appear
    #               once in the sequence, e.g. 1 1 1 1 2 0 3 3 3 1 3 -> 1 2 0 3 1 3
    on, off = phone_info['start'], phone_info['stop']
    phone_feats = raw_representation(on, off, utt_times, utt_feats)
    dom_seq = phone_feats.argmax(axis=1)  # we ignore possible issues with ties here
    if fuse:
        dom_seq = fuse(dom_seq)
    return tuple(dom_seq)  # tuple to be hashable


def dominant_unit_at_center(phone_info, utt_times, utt_feats):
    # dominant unit at frame closest to center time of phone
    phone_dur = phone_info['stop']-phone_info['start']
    phone_center = phone_info['start'] + phone_dur/2.
    on, off = phone_center-.005, phone_center+.005  # get at least 1, at most two samples
    phone_times, phone_feats = raw_representation(on, off, utt_times, utt_feats, return_times=True)
    assert (len(phone_times) == 1) or (len(phone_times) == 2)
    # if 2 samples we arbitrarily take the earliest one
    dom_unit = phone_feats[0,:].argmax() # we ignore possible issues with ties here for the argmax
    return dom_unit


def dominant_unit_around_center(dur, phone_info, utt_times, utt_feats, fuse=False):
    # dominant unit sequences for frame at +/- dur ms from center time of phone
    # convert dur from ms to s
    dur = dur/1000.
    phone_dur = phone_info['stop']-phone_info['start']
    phone_center = phone_info['start'] + phone_dur/2.
    on, off = phone_center-dur, phone_center+dur
    phone_feats = raw_representation(on, off, utt_times, utt_feats)
    dom_seq = phone_feats.argmax(axis=1) # we ignore possible issues with ties here for the argmax
    if fuse:
        dom_seq = fuse(dom_seq)
    return tuple(dom_seq)